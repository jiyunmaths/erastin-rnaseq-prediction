#!/usr/bin/env python3
"""Run extraction, nested comparison, and reporting sequentially with durable status."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    args = ap.parse_args()
    ws = args.workspace.resolve()
    out = Path(__file__).resolve().parent
    work = ws / 'work/bulkformer'
    lock = (work / 'pipeline.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(work / 'packages') + ':' + str(work / 'source'),
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR=str(work / 'matplotlib_cache'), PYTHONUNBUFFERED='1')
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    def status(phase, **extra):
        data = {'phase': phase, 'pid': os.getpid(), 'started_utc': started,
                'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
        p = out / 'pipeline_status.json'
        tmp = p.with_suffix('.tmp');tmp.write_text(json.dumps(data, indent=2) + '\n');tmp.replace(p)
        print(json.dumps(data), flush=True)
    try:
        status('extracting_embeddings')
        subprocess.run([sys.executable, str(out / 'extract_embeddings.py'),
            '--input-dir', str(ws / 'outputs/erastin_40gene_comparison'),
            '--source-dir', str(work / 'source'), '--asset-dir', str(work / 'assets'),
            '--checkpoint-dir', str(work / 'extraction'), '--output-dir', str(out),
            '--threads', '6'], env=env, cwd=ws, check=True)
        status('running_comparison')
        # Ridge and uncertainty calculations do not benefit from 6-way BLAS
        # on these small matrices, and should not compete with other work.
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable, str(out / 'compare_models.py'),
            '--input-dir', str(ws / 'outputs/erastin_40gene_comparison'),
            '--output-dir', str(out)], env=env, cwd=ws, check=True)
        status('complete')
    except BaseException as e:
        status('failed', error=repr(e), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
