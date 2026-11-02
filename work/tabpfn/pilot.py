"""CPU-only timing pilot; no validation response is accessed or scored."""
import gc
import json
import os
import resource
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'work/tabpfn'
for key, value in {
    'SKB_DATA_DIRECTORY': str(WORK/'skrub_data'),
    'TABPFN_MODEL_CACHE_DIR': str(WORK/'checkpoints'),
    'HF_HOME': str(WORK/'hf_cache'),
    'HF_HUB_OFFLINE': '1',
    'TABPFN_NO_BROWSER': '1',
    'TABPFN_DISABLE_TELEMETRY': '1',
    'MPLCONFIGDIR': str(WORK/'matplotlib_cache'),
}.items():
    os.environ[key] = value

import numpy as np
import torch
import tabpfn
from tabpfn import TabPFNRegressor

torch.set_num_threads(3)
torch.set_num_interop_threads(1)
a = np.load(ROOT/'outputs/erastin_elastic_net_comparison/panel_inputs.npz')
s = np.load(ROOT/'outputs/erastin_elastic_net_comparison/shared_splits.npz')
fold = s['inner_fold'][0]
tr = np.flatnonzero((fold == -2)|((fold >= 0)&(fold != 0)))
te = np.flatnonzero(fold == 0)
results = []
for name in ['tabpfn-v3.5-fast-20260909.safetensors', 'tabpfn-v2-regressor-v2_default.ckpt']:
    begin = time.monotonic()
    try:
        model = TabPFNRegressor(model_path=str(WORK/'checkpoints'/name), device='cpu',
                                n_estimators=1, n_preprocessing_jobs=1, random_state=0,
                                categorical_features_indices=[], fit_mode='fit_preprocessors',
                                inference_precision=torch.float32, show_progress_bar=False)
        print('PILOT_START', name, flush=True)
        model.fit(a['x_64'][tr], a['y'][tr])
        fit_seconds = time.monotonic()-begin
        start = time.monotonic()
        pred = model.predict(a['x_64'][te])
        seconds = time.monotonic()-start
        assert pred.shape == (len(te),) and np.isfinite(pred).all()
        result = dict(checkpoint=name, passed=True, train_n=len(tr), predict_n=len(te),
                      fit_seconds=fit_seconds, predict_seconds=seconds, cpu_threads=3,
                      peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      no_accuracy_computed=True, n_estimators=1)
        print('PILOT_COMPLETE', json.dumps(result), flush=True)
        del model, pred
        gc.collect()
    except Exception as exc:
        result = dict(checkpoint=name, passed=False, error=str(exc))
        traceback.print_exc()
    results.append(result)
    (WORK/'pilot_results.json').write_text(json.dumps(dict(tabpfn=tabpfn.__version__, torch=torch.__version__, results=results), indent=2)+'\n')
