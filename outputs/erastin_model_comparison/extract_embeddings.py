#!/usr/bin/env python3
"""Frozen original BulkFormer-147M extraction; resumable, CPU float32."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def projections_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if 'projection_matrix' in name:
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--asset-dir', type=Path, required=True)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    from utils.BulkFormer import BulkFormer
    from model.config import model_params
    from torch_sparse import SparseTensor

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261031)
    cfg = json.loads((args.input_dir / 'comparison_config.json').read_text())
    for item in json.loads((args.input_dir / 'artifact_manifest.json').read_text())['files']:
        if item['path'] in ['comparison_inputs.npz', 'comparison_config.json', 'splits.json']:
            assert sha(args.input_dir / item['path']) == item['sha256']
    d = np.load(args.input_dir / 'comparison_inputs.npz', allow_pickle=False)
    x = d['X_bulkformer_lntpm1']
    ids, pidx, observed = d['sample_ids'], d['panel_indices_0based'], d['bulkformer_observed_gene_mask']
    n = len(ids)
    assert x.shape == (n, 20010) and len(pidx) == 40
    mask_fraction = float((~observed).mean())
    assert mask_fraction == cfg['bulkformer']['mask_prob_argument']
    asset_names = ['BulkFormer_147M.pt', 'G_tcga.pt', 'G_tcga_weight.pt', 'esm2_feature_concat.pt']
    resource_hashes = {p: sha(args.asset_dir / p) for p in asset_names}
    source_hashes = {p: sha(args.source_dir / p) for p in ['utils/BulkFormer.py', 'utils/BulkFormer_block.py', 'utils/Rope.py', 'model/config.py']}
    fingerprint = {'input_sha256': sha(args.input_dir / 'comparison_inputs.npz'),
                   'checkpoint_and_asset_sha256': resource_hashes,
                   'source_sha256': source_hashes, 'dtype': 'float32', 'threads': args.threads,
                   'panel_indices': pidx.tolist(), 'mask_prob_argument': mask_fraction}
    progress_path = args.checkpoint_dir / 'progress.json'
    completed = 0
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        assert previous['fingerprint'] == fingerprint, 'Refusing to mix extraction runs'
        completed = previous['completed']

    graph_indices = torch.load(args.asset_dir / 'G_tcga.pt', map_location='cpu', weights_only=True)
    graph_weights = torch.load(args.asset_dir / 'G_tcga_weight.pt', map_location='cpu', weights_only=True)
    assert graph_indices.shape[0] == 2 and graph_indices.shape[1] == len(graph_weights)
    assert graph_indices.min() >= 0 and graph_indices.max() < 20010
    assert torch.isfinite(graph_weights).all()
    # Exactly the adjacency construction in the authors' pinned notebook.
    graph = SparseTensor(row=graph_indices[1], col=graph_indices[0], value=graph_weights,
                         sparse_sizes=(20010, 20010)).t()
    esm = torch.load(args.asset_dir / 'esm2_feature_concat.pt', map_location='cpu', weights_only=True)
    model = BulkFormer(**model_params, graph=graph, gene_emb=esm)
    state = torch.load(args.asset_dir / 'BulkFormer_147M.pt', map_location='cpu', weights_only=True)
    load_status = model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state, esm
    model.eval().requires_grad_(False)
    assert not any(m.training for m in model.modules())
    initial_projection_hash = projections_digest(model)
    outputs = {}
    for name, shape in [('panel', (n, 643)), ('all_gene', (n, 643)),
                        ('observed_gene', (n, 643)), ('panel_gene_tokens', (n, 40, 643))]:
        path = args.checkpoint_dir / (name + '.npy')
        outputs[name] = np.lib.format.open_memmap(path, mode='r+' if completed else 'w+', dtype=np.float32, shape=shape)
        if completed:
            assert np.isfinite(outputs[name][:completed]).all()
        else:
            outputs[name][:] = np.nan
    print(json.dumps({'event': 'model_loaded', 'state_dict': str(load_status),
        'parameter_count': sum(p.numel() for p in model.parameters()), 'resume_at': completed,
        'total': n, 'device': 'cpu', 'threads': args.threads}), flush=True)
    run_start = time.time()
    start_row = completed
    per_row_seconds = []
    first_full = None
    # Single-sample batches keep CPU memory bounded and remove batch-composition effects.
    with torch.inference_mode():
        for i in range(completed, n):
            t = time.time()
            z = model(torch.from_numpy(x[i:i+1]), mask_prob=mask_fraction, output_expr=False)
            assert z.shape == (1, 20010, 643) and torch.isfinite(z).all(), ids[i]
            a = z[0].numpy()
            outputs['panel'][i] = a[pidx].mean(axis=0)
            outputs['all_gene'][i] = a.mean(axis=0)
            outputs['observed_gene'][i] = a[observed].mean(axis=0)
            outputs['panel_gene_tokens'][i] = a[pidx]
            if i == start_row:
                first_full = a.copy()
            dt = time.time() - t
            per_row_seconds.append(dt)
            # Flush data before publishing the durable completed-row count.
            for value in outputs.values():
                value.flush()
            elapsed = time.time() - run_start
            rate = elapsed / (i + 1 - start_row)
            progress = {'status': 'extracting', 'completed': i + 1, 'total': n,
                'last_ModelID': str(ids[i]), 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'elapsed_this_run_seconds': elapsed, 'seconds_per_sample': rate,
                'estimated_remaining_seconds': rate * (n - i - 1), 'fingerprint': fingerprint}
            atomic_json(progress_path, progress)
            atomic_json(args.output_dir / 'extraction_progress.json', {k:v for k,v in progress.items() if k != 'fingerprint'})
            if i % 5 == 0 or i + 1 == n:
                print(json.dumps({k:v for k,v in progress.items() if k != 'fingerprint'}), flush=True)
            del z, a
        # Check that inference has not changed the fixed random attention projections.
        assert projections_digest(model) == initial_projection_hash
        repeat_row = start_row if start_row < n else 0
        repeated = model(torch.from_numpy(x[repeat_row:repeat_row+1]), mask_prob=mask_fraction, output_expr=False)[0].numpy()
        if first_full is not None:
            repeat_max_abs = float(np.max(np.abs(repeated - first_full)))
            assert np.array_equal(repeated, first_full), 'Frozen CPU inference was not deterministic'
        else:
            repeat_max_abs = float(np.max(np.abs(repeated[pidx].mean(0) - outputs['panel'][repeat_row])))
            assert repeat_max_abs <= 1e-6
    for value in outputs.values():
        assert np.isfinite(value).all()
    np.savez_compressed(args.output_dir / 'bulkformer_embeddings.npz', sample_ids=ids,
        panel_embedding=np.asarray(outputs['panel']), all_gene_embedding=np.asarray(outputs['all_gene']),
        observed_gene_embedding=np.asarray(outputs['observed_gene']), panel_indices_0based=pidx,
        panel_symbols=d['panel_symbols'])
    np.savez_compressed(args.output_dir / 'panel_gene_embeddings.npz', sample_ids=ids,
        panel_symbols=d['panel_symbols'], embeddings=np.asarray(outputs['panel_gene_tokens']))
    metadata = {'status': 'complete', 'sample_count': n, 'embedding_dimension': 643,
        'checkpoint_family_label': 'authors original BulkFormer-147M',
        'actual_parameter_count': sum(p.numel() for p in model.parameters()),
        'strict_state_dict_loading': True, 'graph_edges': len(graph_weights),
        'single_sample_batch_size': 1, 'no_autocast': True, 'device': 'cpu',
        'frozen_attention_projection_sha256': initial_projection_hash,
        'deterministic_repeat_max_absolute_error': repeat_max_abs,
        'elapsed_this_run_seconds': time.time() - run_start, 'resumed_from_row': start_row,
        'fingerprint': fingerprint, 'versions': {p:importlib.metadata.version(p)
            for p in ['torch','torch-geometric','torch-sparse','torch-scatter','performer-pytorch','numpy']},
        'notes': ['Full transcriptome fed to all embedding arms.',
                  'ESM resource supplied to the original constructor but not used by this pinned forward implementation.',
                  'Checkpoint includes its trained/fixed Performer random projections; none redrawn.',
                  'Full all-gene pooling includes missing-gene token outputs.',
                  'Floating point reductions follow the authors NumPy pooling convention.']}
    atomic_json(args.output_dir / 'embedding_metadata.json', metadata)
    atomic_json(args.output_dir / 'extraction_progress.json', {'status': 'complete', 'completed': n, 'total': n})
    print(json.dumps({'event': 'extraction_complete', 'samples': n}), flush=True)


if __name__ == '__main__':
    main()
