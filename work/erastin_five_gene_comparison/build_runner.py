from pathlib import Path
import shutil
out=Path('outputs/erastin_five_gene_comparison');src=Path('outputs/erastin_stratified_panel_comparison')
for name in ['model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']:shutil.copyfile(src/name,out/name)
s=(src/'run_panels.py').read_text()
s=s.replace('Unweighted 40/64/80-gene ridge, NN and XGBoost on shared stratified splits.','Exploratory fixed five-gene versus 64-gene ridge/NN comparison on shared splits.')
s=s.replace('import joblib,numpy as np,scipy,sklearn,xgboost as xgb','import joblib,numpy as np,scipy,sklearn')
s=s.replace('import xgb_helpers as xp\n','')
s=s.replace("FAMILIES=['ridge','neural_network','xgboost'];PANELS=[40,64,80]", "FAMILIES=['ridge','neural_network'];PANELS=[5,64]")
s=s.replace('sys.dont_write_bytecode=True','FIVE_GENES=["SLC7A11","GPX4","ABCC1","ACSL4","AIFM2"]\nsys.dont_write_bytecode=True')
s=s.replace("def matrix(x,genes,y=None):return xgb.DMatrix(x,label=y,feature_names=list(genes),nthread=1)\n",'')
s=s.replace("else STATE['nn_grid'] if f=='neural_network' else xp.GRID", "else STATE['nn_grid']")
a=s.index('        else:\n            da=matrix(');b=s.index('    assert np.isfinite(val',a);s=s[:a]+s[b:]
a=s.index('    else:\n        mu,sd=float(y[tr]');b=s.index('    arrays=dict(',a)
s=s[:a]+"    assert np.isfinite(p).all();joblib.dump(saved,root.with_suffix('.joblib'),compress=3)\n"+s[b:]
s=s.replace("    m[ext+'_sha256']=sha(root.with_suffix('.'+ext))\n    if family=='xgboost':m.update(saved)","    m['joblib_sha256']=sha(root.with_suffix('.joblib'))")
s=s.replace("['npz','ubj' if family=='xgboost' else 'joblib']","['npz','joblib']")
a=s.index('def prepare():');b=s.index('\ndef main():',a)
s=s[:a]+'''def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);sources=[];reuse=[]
    source=WS/'outputs/erastin_stratified_panel_comparison';known={v['path']:v['sha256'] for v in json.loads((source/'artifact_manifest.json').read_text())['files']}
    names=['panel_inputs.npz','shared_splits.npz','protocol.json','predictions.npz','summary.tsv','validation.json','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv','model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']
    for name in names:
        assert sha(source/name)==known[name];sources.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    selection=WS/'outputs/erastin_stratified_gene_importance';sknown={v['path']:v['sha256'] for v in json.loads((selection/'artifact_manifest.json').read_text())['files']}
    for name in ['protocol.json','summary.json','gene_importance_summary.tsv']:
        assert sha(selection/name)==sknown[name];sources.append(dict(path=str((selection/name).relative_to(WS)),sha256=sknown[name]))
    oldp=json.loads((source/'protocol.json').read_text());a=np.load(source/'panel_inputs.npz');spl=np.load(source/'shared_splits.npz');genes=list(a['genes_64']);indices=np.array([genes.index(g) for g in FIVE_GENES]);assert len(set(indices))==5
    arrays=dict(x_5=a['x_64'][:,indices],x_64=a['x_64'],genes_5=np.array(FIVE_GENES),genes_64=a['genes_64'],five_gene_indices_in_64=indices)
    for key in ['y','sample_ids','group_ids','lineage']:arrays[key]=a[key]
    path=OUT/'panel_inputs.npz'
    if path.exists():
        z=np.load(path)
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(path,**arrays)
    for name in ['shared_splits.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:
        dest=OUT/name
        if dest.exists():assert sha(dest)==sha(source/name)
        else:shutil.copyfile(source/name,dest)
    for name in ['model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']:assert sha(OUT/name)==sha(source/name)
    for family in FAMILIES:
        for r in range(20):
            for ext in ['npz','joblib','json']:
                rel=f'folds/{family}__p64__r{r+1:02d}.{ext}';assert sha(source/rel)==known[rel];sources.append(dict(path=str((source/rel).relative_to(WS)),sha256=known[rel]))
                dest=OUT/('source_metadata' if ext=='json' else 'folds')/f'{family}__p64__r{r+1:02d}.{ext}'
                if dest.exists():assert sha(dest)==known[rel]
                else:shutil.copyfile(source/rel,dest)
                reuse.append(dict(source=str((source/rel).relative_to(WS)),local=str(dest.relative_to(OUT)),sha256=known[rel]))
    p=dict(status='frozen_before_fixed_five_gene_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=744,identity_groups=733,lineages=23,excluded_samples=oldp['excluded_samples'],
        panels={'5':FIVE_GENES,'64':genes},five_gene_indices_in_64=indices,split_design=oldp['split_design'],inner_design=oldp['inner_design'],ampulla_handling=oldp['ampulla_handling'],
        outer_train_sizes=oldp['outer_train_sizes'],outer_test_sizes=oldp['outer_test_sizes'],split_seeds=spl['outer_split_seeds'],inner_split_seeds=spl['inner_split_seeds'],neural_seed_triplets=spl['neural_initialization_seeds'],
        ridge_alphas=oldp['ridge_alphas'],neural_grid=oldp['neural_grid'],neural_fixed=oldp['neural_fixed'],weighting=oldp['weighting'],tuning=oldp['tuning'],
        scaling='Training-only unweighted StandardScaler for ridge/NN features, target centering for ridge and training-only y standardization for NN.',
        controls='Byte-identical shared outer/inner split arrays and seed triplets. All other model settings and grids unchanged. Both panels separately tuned using inner training/validation only.',
        baseline='Reuse 20 verified 64-gene ridge and 20 verified 64-gene NN models. Fit both five-gene models from scratch on each outer training set, with separate inner hyperparameter selection; 40 new + 40 reused outer models.',
        feature_selection='The five genes were explicitly fixed by the user after prior held-out importance analysis of the same 20 splits. They are NOT selected independently within each outer training set.',
        selection_limitation='Earlier test responses influenced the choice of this five-gene panel. Reusing the same outer splits gives an exploratory post-selection comparison, not independent validation of feature selection. Inner hyperparameter tuning does not remove this selection bias; a fresh response cohort or prospectively reserved test set is required for confirmatory validation.',
        primary_comparisons=[['ridge__5','ridge__64'],['neural_network__5','neural_network__64']],
        secondary_comparisons=[['neural_network__5','ridge__5'],['neural_network__64','ridge__64'],['neural_network__5','ridge__64'],['ridge__5','neural_network__64']],
        primary_metrics=oldp['primary_metrics'],secondary_metrics=oldp['secondary_metrics'],metric_notes=oldp['metric_notes'],
        uncertainty=dict(count=5000,seed=2026110901,unit=oldp['uncertainty']['unit']+' These conditional intervals also do not adjust for the prior five-gene selection.'),
        equivalence='No equivalence/noninferiority margin or confirmatory equivalence test is defined. An interval crossing zero does not establish equivalence.',
        limitations=oldp['limitations'],new_outer_models=40,reused_outer_models=40,total_outer_models=80,workers=WORKERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz']},
        scripts={name:sha(OUT/name) for name in ['run_comparison.py','model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py','validate_preflight.py']})
    path=OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());p['created_utc']=existing['created_utc'];assert existing==clean(p)
    else:dump(path,p)
    ph=sha(path);dump(OUT/'source_manifest.json',dict(files=sources));dump(OUT/'reuse_manifest.json',dict(files=reuse,source_protocol_sha256=sha(source/'protocol.json')))
    for family in FAMILIES:
        for r in range(20):
            original=json.loads((OUT/'source_metadata'/f'{family}__p64__r{r+1:02d}.json').read_text());assert original['family']==family and original['panel']==64 and original['protocol_sha256']==sha(source/'protocol.json')
            m=dict(original,reused=True,protocol_sha256=ph,reuse_source_protocol_sha256=original['protocol_sha256'],original_metadata_sha256=sha(OUT/'source_metadata'/f'{family}__p64__r{r+1:02d}.json'))
            dest=OUT/'folds'/f'{family}__p64__r{r+1:02d}.json'
            if dest.exists():assert json.loads(dest.read_text())==clean(m)
            else:dump(dest,m)
    return ph
''' + s[b:]
s=s.replace("if not(p==64 and f=='neural_network')","if p==5")
s=s.replace('total=160,reused=20','total=40,reused=40').replace('(160-n)','(40-n)')
s=s.replace('new_outer_models=160,reused_outer_models=20,total_outer_models=180','new_outer_models=40,reused_outer_models=40,total_outer_models=80')
s=s.replace("completed=160,total=160,reused=20","completed=40,total=40,reused=40")
(out/'run_comparison.py').write_text(s)
