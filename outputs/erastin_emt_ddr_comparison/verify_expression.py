"""Independently re-extract every expanded feature from the original CSV."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from model_helpers import sha,dump
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
def main():
    begin=time.monotonic();raw=WS/'work/erastin/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv';manifest=json.load(open(WS/'outputs/erastin_40gene_comparison/source_manifest.json'));expected=next(r['sha256'] for r in manifest['files'] if r['local_name']==raw.name);assert sha(raw)==expected
    data=np.load(OUT/'panel_inputs.npz');panel='ferro64_emt_ddr';columns=[f'{symbol} ({ent})' for symbol,ent in zip(data['genes_'+panel],data['entrez_'+panel])];assert len(set(columns))==413 and 'BOLA2 (552900)' in columns and 'BOLA2B (552900)' not in columns;first=pd.read_csv(raw,nrows=0).columns[0];frame=pd.read_csv(raw,usecols=[first]+columns,index_col=0).loc[data['sample_ids'],columns];x=frame.to_numpy(np.float32).astype(float);np.testing.assert_array_equal(x,data['x_'+panel]);assert np.isfinite(x).all()
    for p in ['ferro5','ferro64','ferro64_emt','ferro64_ddr']:
        lookup={c:j for j,c in enumerate(columns)};indices=[lookup[f'{g} ({e})'] for g,e in zip(data['genes_'+p],data['entrez_'+p])];np.testing.assert_array_equal(x[:,indices],data['x_'+p])
    dump(OUT/'raw_expression_validation.json',dict(passed=True,source_path=str(raw.relative_to(WS)),source_sha256=expected,samples=744,unique_expanded_features=413,entries_checked=744*413,max_error=0.,baseline_and_every_expanded_panel_match=True,precision='Original float32 storage convention retained exactly, then promoted to float64 for fitting.',BOLA2_column_verified=True,seconds=time.monotonic()-begin));print('RAW_EXPRESSION_VERIFIED',flush=True)
if __name__=='__main__':main()
