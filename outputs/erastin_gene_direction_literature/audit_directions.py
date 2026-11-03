#!/usr/bin/env python3
"""Verify signed effects in the retained models and compare curated literature."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
SOURCE=WS/'outputs/erastin_prism_ctrp_gene_importance'
REGISTRY=WS/'outputs/retained_64_gene_models/model_registry.json'
sys.path.insert(0,str(SOURCE))
sys.dont_write_bytecode=True
FAMILIES=['neural_network','ridge','elastic_net']
NAMES={'neural_network':'NN','ridge':'Ridge','elastic_net':'ElasticNet'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def gradient(model,x):
    z=model['x_scaler'].transform(x);g=np.zeros_like(z)
    for member in model['members']:
        h=np.tanh(z@member.coefs_[0]+member.intercepts_[0])
        g+=((1-h*h)*member.coefs_[1].ravel())@member.coefs_[0].T
    return g*model['y_scaler'].scale_[0]/len(model['members'])


def assess(sign,stable,selected,e):
    if selected==0:return 'not_selected'
    if not stable:return 'unstable_model_sign'
    if e['literature_role']=='M':return 'context_dependent_literature'
    if e['literature_role']=='U':return 'unresolved_literature'
    match=sign==e['expected_sensitivity_sign']
    prefix='same_direction' if match else 'opposite_direction'
    if e['evidence_level']=='pathway':return prefix+'_pathway_inference'
    if e['evidence_level']=='isoform':return prefix+'_isoform_caution'
    if e['inducer_scope']=='other_inducer':return prefix+'_other_inducer'
    return prefix+'_functional_evidence'


def main():
    records={}
    def record(p,expected=None):
        h=sha(p)
        if expected is not None:assert h==expected,str(p)
        key=str(Path(p).relative_to(WS));records[key]=dict(path=key,sha256=h)
    record(REGISTRY);record(SOURCE/'artifact_manifest.json')
    manifest={r['path']:r['sha256'] for r in json.load(open(SOURCE/'artifact_manifest.json'))['files']}
    for n in ['source_manifest.json','gene_importance_summary.tsv','coefficient_sensitivity_by_repeat.tsv',
              'coefficient_selection.tsv','validation.json','ctrp_importance_arrays.npz','prism_importance_arrays.npz',
              'ctrp_inputs.npz','prism_inputs.npz','weighted_mlp.py']:
        record(SOURCE/n,manifest[n])
    originals={r['path']:r['sha256'] for r in json.load(open(SOURCE/'source_manifest.json'))['files']}
    assert json.load(open(SOURCE/'validation.json'))['passed']
    registry=json.load(open(REGISTRY));genes=np.array(registry['genes_in_input_order'])
    lit=json.load(open(OUT/'candidate_gene_evidence.json'))
    evidence={r['gene']:r for r in lit['genes']};refs=json.load(open(OUT/'references.json'))
    assert list(evidence)==list(genes) and len(genes)==64
    source_summary=pd.read_csv(SOURCE/'gene_importance_summary.tsv',sep='\t')
    repeat_ref=pd.read_csv(SOURCE/'coefficient_sensitivity_by_repeat.tsv',sep='\t')
    selections=pd.read_csv(SOURCE/'coefficient_selection.tsv',sep='\t')
    rows=[];perrepeat=[];perlineage=[];checks=[];maxerror=0.;maxgradient=0.
    for ds,cohort in registry['cohorts'].items():
        inp=np.load(SOURCE/(ds.lower()+'_inputs.npz'));arr=np.load(SOURCE/(ds.lower()+'_importance_arrays.npz'))
        np.testing.assert_array_equal(inp['genes'],genes)
        for name in ['panel_inputs.npz','shared_splits.npz','protocol.json']:
            file=cohort['input_and_validation_files'][name];record(WS/file['path'],file['sha256'])
        current=np.load(WS/cohort['input_and_validation_files']['panel_inputs.npz']['path'])
        for old,new in [('x','x_64'),('y','y'),('genes','genes_64'),('ids','sample_ids'),('groups','group_ids'),('lineage','lineage')]:
            np.testing.assert_array_equal(inp[old],current[new])
        sp=np.load(WS/cohort['input_and_validation_files']['shared_splits.npz']['path'])
        np.testing.assert_array_equal(inp['test'],sp['outer_test'])
        for family in FAMILIES:
            mm=['ridge','elastic_net','neural_network'].index(family)
            slopes=[];grads=[]
            for entry in cohort['models'][family]['outer_models']:
                r=entry['repeat']-1
                for file in entry['files'].values():
                    assert originals[file['path']]==file['sha256'];record(WS/file['path'],file['sha256'])
                model=joblib.load(WS/entry['files']['joblib']['path'])
                fit=np.load(WS/entry['files']['npz']['path']);te=np.flatnonzero(inp['test'][r]);tr=np.flatnonzero(~inp['test'][r])
                np.testing.assert_array_equal(fit['test_rows'],te);np.testing.assert_array_equal(fit['train_rows'],tr)
                assert set(inp['groups'][tr]).isdisjoint(inp['groups'][te])
                if family=='ridge':beta=model['coef'];g=np.broadcast_to(beta,(len(te),64))
                elif family=='elastic_net':
                    beta=model['model'].coef_*model['y_scaler'].scale_[0];g=np.broadcast_to(beta,(len(te),64))
                else:
                    g=gradient(model,inp['x'][te]);beta=g.mean(axis=0)
                reference=repeat_ref[(repeat_ref.dataset==ds)&(repeat_ref.model==family)&(repeat_ref['repeat']==r+1)]
                ref=reference.set_index('gene').loc[genes].mean_standardized_slope.to_numpy()
                error=float(np.max(np.abs(beta-ref)));maxerror=max(maxerror,error)
                np.testing.assert_allclose(beta,ref,rtol=1e-9,atol=1e-12)
                expected=arr['G'][mm,r]
                for li,label in enumerate(arr['labels']):
                    actual=g[inp['lineage'][te]==label].mean(axis=0)
                    error=float(np.max(np.abs(actual-expected[li])));maxgradient=max(maxgradient,error)
                    assert error<1e-12
                slopes.append(beta);grads.append(g)
                for j,gene in enumerate(genes):perrepeat.append(dict(dataset=ds,model=family,repeat=r+1,gene=str(gene),signed_standardized_slope=float(beta[j])))
                checks.append(dict(dataset=ds,model=family,repeat=r+1,model_verified=True))
            slopes=np.array(slopes);g=np.concatenate(grads);mean=slopes.mean(axis=0)
            native=source_summary[(source_summary.dataset==ds)&(source_summary.model==family)&(source_summary.scope=='native_pooled')].set_index('gene').loc[genes]
            np.testing.assert_allclose(mean,native.mean_standardized_slope,rtol=1e-9,atol=1e-12)
            eligible=arr['cohort_n']>=10
            lineage_effect=arr['G'][mm].mean(axis=0)
            for j,gene in enumerate(genes):
                values=slopes[:,j];pos=int((values>0).sum());neg=int((values<0).sum());zero=int((values==0).sum())
                sign=int(np.sign(mean[j]));match=pos if sign>0 else neg if sign<0 else zero
                selected=20-zero
                # Cell-weighted test appearances, not independent cell-level replications.
                agreement=float(np.mean(g[:,j]>0) if sign>0 else np.mean(g[:,j]<0) if sign<0 else np.mean(g[:,j]==0))
                stable=bool(sign!=0 and match>=18 and (family!='neural_network' or agreement>=.9-1e-12))
                if family=='elastic_net':
                    sel=selections[(selections.dataset==ds)&(selections.model==family)&(selections.gene==gene)].iloc[0]
                    assert (selected,pos,neg)==(sel.selected_fits,sel.positive_fits,sel.negative_fits)
                e=evidence[str(gene)];n=native.loc[gene]
                counts=int((np.sign(lineage_effect[eligible,j])==sign).sum())
                rows.append(dict(dataset=ds,model=family,gene=str(gene),mean_standardized_slope=float(mean[j]),
                    mean_sign=sign,higher_expression_predicts='sensitivity' if sign>0 else 'resistance' if sign<0 else 'no_mean_effect',
                    positive_fits=pos,negative_fits=neg,zero_fits=zero,selected_fits=selected,same_direction_fits=match,
                    gradient_or_coefficient_direction_agreement=agreement,stable_sign=stable,
                    matching_large_lineages=counts,eligible_large_lineages=int(eligible.sum()),
                    predictive_rank=int(n['rank']),relative_mse_increase_pct=float(n.relative_mse_increase_percent),
                    relative_importance_ci_low=float(n.relative_ci_low),relative_importance_ci_high=float(n.relative_ci_high),
                    positive_importance_splits=int(n.positive_repeats),
                    literature_role=e['literature_role'],evidence_level=e['evidence_level'],inducer_scope=e['inducer_scope'],
                    literature_assessment=assess(sign,stable,selected,e),note=e['note'],reference_ids=';'.join(e['reference_ids']),reference_urls=';'.join(e['reference_urls'])))
                for li,label in enumerate(arr['labels']):perlineage.append(dict(dataset=ds,model=family,gene=str(gene),lineage=str(label),
                    cohort_n=int(arr['cohort_n'][li]),mean_standardized_slope=float(lineage_effect[li,j])))
    long=pd.DataFrame(rows)
    assert long.shape[0]==384
    long.to_csv(OUT/'model_directions_long.tsv',sep='\t',index=False,float_format='%.12g')
    pd.DataFrame(perrepeat).to_csv(OUT/'per_split_signed_effects.tsv',sep='\t',index=False,float_format='%.12g')
    pd.DataFrame(perlineage).to_csv(OUT/'per_lineage_signed_effects.tsv',sep='\t',index=False,float_format='%.12g')
    wide=[]
    for gene in genes:
        e=evidence[str(gene)];row=dict(gene=str(gene),literature_role=e['literature_role'],evidence_level=e['evidence_level'],inducer_scope=e['inducer_scope'])
        for ds in ['CTRP','PRISM']:
            for family in FAMILIES:
                d=long[(long.dataset==ds)&(long.model==family)&(long.gene==gene)].iloc[0]
                prefix=ds+'_'+NAMES[family]+'_'
                for col in ['mean_sign','higher_expression_predicts','mean_standardized_slope','same_direction_fits','selected_fits','stable_sign',
                            'literature_assessment','gradient_or_coefficient_direction_agreement','predictive_rank',
                            'relative_mse_increase_pct','relative_importance_ci_low','relative_importance_ci_high']:
                    row[prefix+col]=d[col]
        row.update(note=e['note'],references=';'.join(e['reference_urls']));wide.append(row)
    wide=pd.DataFrame(wide);wide.to_csv(OUT/'summary.tsv',sep='\t',index=False,float_format='%.12g')
    wide.to_csv(OUT/'all64_gene_signs_and_literature.csv',index=False,float_format='%.12g')
    compact_cols=['gene','literature_role','evidence_level','inducer_scope']
    for ds in ['CTRP','PRISM']:
        compact_cols += [ds+'_NN_'+c for c in ['mean_sign','higher_expression_predicts','same_direction_fits','stable_sign','literature_assessment']]
    compact_cols += ['note','references']
    wide[compact_cols].to_csv(OUT/'nn_gene_direction_comparison.csv',index=False)
    counts=long.groupby(['dataset','model','literature_assessment']).size().rename('genes').reset_index()
    counts.to_csv(OUT/'agreement_counts.tsv',sep='\t',index=False)
    nn=long[long.model=='neural_network']
    result=dict(genes=64,models=120,assay_model_gene_rows=384,
        stable_NN_genes={ds:int(nn[nn.dataset==ds].stable_sign.sum()) for ds in ['CTRP','PRISM']},
        NN_same_sign_across_assays=int(np.sum(wide.CTRP_NN_mean_sign==wide.PRISM_NN_mean_sign)),
        NN_stable_same_sign_across_assays=int(np.sum(wide.CTRP_NN_stable_sign & wide.PRISM_NN_stable_sign & (wide.CTRP_NN_mean_sign==wide.PRISM_NN_mean_sign))),
        literature_role_counts={str(k):int(v) for k,v in wide.literature_role.value_counts().items()},
        NN_comparison_counts=counts[counts.model=='neural_network'].to_dict('records'))
    dump('summary.json',result)
    figures(long,evidence,genes)
    report(long,wide,evidence,refs,result)
    for n in ['curate_evidence.py','candidate_gene_evidence.json','references.json','audit_directions.py']:record(OUT/n)
    panel=WS/'outputs/ferroptosis_paper_gene_panel.json';record(panel)
    pdf=Path(json.load(open(panel))['source']['path'])
    expected=json.load(open(panel))['source']['sha256'];assert sha(pdf)==expected
    dump('source_manifest.json',dict(files=list(records.values()),user_supplied_pdf=dict(path=str(pdf),sha256=expected)))
    dump('protocol.json',dict(target='Negative erastin AUC: positive sign predicts higher sensitivity; negative sign predicts higher resistance.',
        models='All 120 exact retained outer models: 20 repeats x 3 families x 2 assays. No fitting or panel changes.',
        signed_measure='Ridge coefficient, Elastic Net coefficient times training target SD, or mean ensemble derivative with respect to standardized expression. Units are original target units per current-training expression SD.',
        stability='Descriptive >=18/20 nonzero fitted mean slopes in the same direction. NN additionally requires >=90% of held-out test appearances to have matching ensemble-gradient sign. Exact EN zeros stay unselected. Not a statistical significance threshold.',
        biological_comparison='Functional activity expectations are not universal basal RNA coefficients. Separate functional evidence, pathway inference, other-inducer evidence, LAMP2A isoform evidence, mixed roles, unresolved CTSD citation conflict, and unstable/unselected model signs.',
        search='Targeted current literature search anchored to supplied 2024 review and primary studies; not a systematic review. Functional rather than exclusively expression-prediction evidence. Analyst not blinded to previously reviewed model results.',
        sources_checked='2026-11-02'))
    dump('validation.json',dict(
        passed=True,genes=64,models=120,assay_model_gene_rows=384,model_checks=checks,
        source_hashes_verified=True,registry_gene_order_exact=True,no_refitting=True,no_panel_change=True,
        exact_elastic_net_zero_and_sign_counts=True,max_saved_repeat_slope_error=maxerror,max_lineage_gradient_error=maxgradient,
        sign_definition_verified=True,literature_gene_coverage=64,literature_references=len(refs),
        CTSD_CTSB_source_discrepancy_flagged=True,NN_GPX4_PRISM_unstable=not bool(wide.loc[wide.gene=='GPX4','PRISM_NN_stable_sign'].iloc[0])))
    # Seal user-facing analysis without altering any prior experiment.
    dump('artifact_manifest.json',dict(files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']))
    print(json.dumps(result,indent=2),flush=True)


def figures(long,evidence,genes):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    columns=[('Literature',None,None)]+[(ds+'\n'+NAMES[f],ds,f) for ds in ['CTRP','PRISM'] for f in FAMILIES]
    colors=np.ones((64,7,4));annotations=np.empty((64,7),object)
    red=np.array([.76,.23,.20,1.]);blue=np.array([.14,.36,.66,1.]);grey=np.array([.87,.87,.87,1.])
    for i,gene in enumerate(genes):
        e=evidence[str(gene)];sign=e['expected_sensitivity_sign']
        colors[i,0]=red if sign>0 else blue if sign<0 else grey
        annotations[i,0]={'S':'+','R':'-','M':'M','U':'?'}[e['literature_role']]
        if e['evidence_level']=='pathway':annotations[i,0]+='p'
        if e['inducer_scope']=='other_inducer':annotations[i,0]+='o'
        if e['evidence_level']=='isoform':annotations[i,0]+='i'
        for j,(_,ds,family) in enumerate(columns[1:],1):
            r=long[(long.dataset==ds)&(long.model==family)&(long.gene==gene)].iloc[0]
            base=red if r.mean_sign>0 else blue if r.mean_sign<0 else grey
            colors[i,j]=base if r.stable_sign else .23*base+.77*np.ones(4)
            colors[i,j,3]=1.
            annotations[i,j]=('+' if r.mean_sign>0 else '-')+str(r.same_direction_fits)+'/20' if r.mean_sign!=0 else 'zero 20/20'
    fig,ax=plt.subplots(figsize=(11.2,20.6));ax.imshow(colors,aspect='auto')
    for i in range(64):
        for j in range(7):
            stable=j==0 or (colors[i,j,:3].mean()<.7)
            ax.text(j,i,annotations[i,j],ha='center',va='center',fontsize=7.4,color='white' if stable and annotations[i,j] not in ['M','?'] else '#222222')
    ax.set_yticks(range(64),genes);ax.set_xticks(range(7),[v[0] for v in columns]);ax.xaxis.tick_top();ax.tick_params(length=0,pad=8)
    ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,64,1),minor=True);ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',bottom=False,left=False)
    fig.suptitle('Signed model effects versus ferroptosis literature',fontsize=16,weight='bold',y=.985)
    fig.text(.5,.958,'+ predicts sensitivity; - predicts resistance · target = negative erastin AUC',ha='center',fontsize=11)
    fig.legend(handles=[Patch(facecolor=red,label='+ sensitivity'),Patch(facecolor=blue,label='- resistance'),Patch(facecolor='#eeeeee',label='pale: unstable or unselected model effect')],loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.text(.5,.016,'Model cells: sign of mean effect, followed by matching fits /20. NN uses signed input gradients.\nLiterature suffixes: p = pathway inference, o = other inducer, i = LAMP2A isoform; M = mixed, ? = unresolved.\nFunctional perturbation direction is not a universal baseline RNA coefficient.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.14,right=.98,top=.925,bottom=.08)
    fig.savefig(OUT/'all64_signed_effects.png',dpi=160,facecolor='white');fig.savefig(OUT/'all64_signed_effects.svg',facecolor='white');plt.close(fig)


def report(long,wide,evidence,refs,result):
    refnum={key:i+1 for i,key in enumerate(refs)}
    def cite(ids):return ', '.join(f'[{refnum[i]}]({refs[i]["url"]})' for i in ids)
    def cell(ds,gene):
        r=long[(long.dataset==ds)&(long.model=='neural_network')&(long.gene==gene)].iloc[0]
        return ('+' if r.mean_sign>0 else '-' if r.mean_sign<0 else '0')+f' ({r.same_direction_fits}/20)'+('' if r.stable_sign else ' unstable')
    lines=['# Signed effects of all 64 genes versus ferroptosis literature','',
        '**Positive means higher predicted erastin sensitivity; negative means higher predicted resistance.** Both response targets are negative AUC. This audit covers all 120 retained models, with 20 saved fits per family and assay. Neural-network (NN) signs are ensemble input gradients, not raw first-layer weights. Ridge and Elastic Net use signed standardized-expression coefficients.','',
        'The supplied review and primary studies were checked in a targeted literature audit through 2026-11-02. This is not an exhaustive systematic review. Literature roles describe increased functional activity in the cited contexts; they do not prescribe the sign of a basal mRNA coefficient across cancer lineages.','',
        '## Findings','',
        'The NN directions for **SLC7A11 (-), AIFM2/FSP1 (-), GCLC (-), and ACSL4 (+)** agree with the cited defense/sensitization mechanisms in both assays, with matching signs in all 20 fits. **LRP8 (-)** also agrees with its published resistance role, particularly established in GPX4-inhibitor studies. '+cite(['xc','fsp1','gclc','acsl4','lrp8'])+'.','',
        '**GPX4 is negative in every CTRP NN fit but is unstable in PRISM (10 positive and 10 negative fits).** Its small negative PRISM mean must not be reported as reliable cross-assay agreement. The established protective GPX4 mechanism does not make an unstable model effect informative. '+cite(['gpx4'])+'.','',
        '**POR and CYB5R1 have stable negative NN effects in both assays**, opposite to the pro-ferroptotic oxidoreductase function reported in primary work. This is a useful model-association discrepancy, not evidence that the proteins protect cells experimentally. '+cite(['por'])+'.','',
        '**ABCC1 is a special case:** GSH export can sensitize cells to ferroptosis despite resistance to some other drugs. The CTRP positive sign agrees; the PRISM negative sign opposes this role, although its held-out importance is weak. '+cite(['abcc1'])+'.','',
        '**GGT1 and PROM2** have resistance evidence, but positive PRISM NN slopes. **HDDC3/MESH1** has erastin-sensitization evidence but negative NN slopes. **LAMP2** is negative in both NNs whereas the relevant primary evidence concerns sensitizing LAMP2A-mediated GPX4 degradation; total-gene RNA is an isoform mismatch. '+cite(['ggt1','prom2','mesh1','cma'])+'.','',
        '**DHFR and BH4-pathway annotations need an inducer qualifier.** The cited study shows BH4 dependence and DHFR-mediated protection under GPX4 inhibition, not the same requirement under cystine deprivation. Positive DHFR coefficients oppose that general defense expectation but are not a clean falsification of an erastin-specific prediction. '+cite(['bh4'])+'.','',
        '**CTSD source correction:** the supplied review says cathepsin D in its albumin-catabolism paragraph and cites reference 59. That primary study explicitly identifies cathepsin B (CTSB), not CTSD, as the principal albumin-degrading protease. The earlier panel annotation inherited this source discrepancy. CTSD remains in the retained 64-gene model, but this citation cannot establish its protective direction. A newer CTSD glycosylation study in colorectal metastasis is context/PTM-specific and does not settle the pan-cancer erastin RNA sign. '+cite(['review','ctsb','ctsd2025'])+'.','',
        '**Context-dependent cases are not forced into a binary agreement score:** LPCAT3, ACSL3, AGPS, NFE2L2, ATF4 and GOT1. The cited contexts include different lipid substrates, KRAS-mutant lung cancer, antioxidant efflux programs and starvation versus pancreatic-cancer metabolism. '+cite(['kras','ether_opposite','abcc1','atf4_protect','atf4_promote','glutam','got1'])+'.','',
        '**GSR has recent functional corroboration:** the September 2026 study reports sensitization after GSR knockout, including erastin. It also analyzes CTRP expression-response associations; that observational concordance reuses public cell-line data and must not be counted as independent validation of our prediction results. '+cite(['gsr'])+'.','',
        '## Complete gene-by-gene comparison','',
        'Literature **S** = functional sensitizer; **R** = functional resistance/protection; **M** = documented mixed roles; **?** = unresolved. These codes are not a universal expression biomarker label. Model cells show sign and matching fits out of 20. A stable descriptive label requires at least 18/20 nonzero same-direction fits and, for NN, at least 90% of held-out appearances with matching gradient sign. Pale cells in the figure fail this rule. No significance test is implied.','',
        '| Gene | Literature role | CTRP NN | PRISM NN | Evidence context / caveat | Sources |','|---|---|---|---|---|---|']
    for gene in wide.gene:
        e=evidence[gene];role={'R':'R (-)','S':'S (+)','M':'M','U':'?'}[e['literature_role']]
        label=e['evidence_level']+'; '+e['inducer_scope']
        lines.append(f'| {gene} | {role} | {cell("CTRP",gene)} | {cell("PRISM",gene)} | {label}. {e["note"]} | {cite(e["reference_ids"])} |')
    lines+=['','![All six model/assay sign profiles](all64_signed_effects.png)','','## All-model summaries','',
        'The complete spreadsheet-style table includes the signed standardized effect, repeat support, direction stability, held-out predictive rank/importance and literature assessment for **ridge, Elastic Net and NN** in both assays. Elastic Net zeros are retained as unselected rather than assigned a positive or negative biological direction.','',
        '[Concise NN gene-by-gene comparison (CSV)](nn_gene_direction_comparison.csv) · [All 64 genes, six model/assay combinations (CSV)](all64_gene_signs_and_literature.csv) · [TSV summary](summary.tsv) · [Detailed model effects](model_directions_long.tsv) · [Per-lineage slopes](per_lineage_signed_effects.tsv)','',
        f'NN mean signs are the same across assays for {result["NN_same_sign_across_assays"]}/64 genes; {result["NN_stable_same_sign_across_assays"]}/64 have the same sign and meet the descriptive stability rule in both. The number of stable NN signs is {result["stable_NN_genes"]["CTRP"]} in CTRP and {result["stable_NN_genes"]["PRISM"]} in PRISM. These counts concern model stability, not biological validation.','',
        '## Interpretation','',
        'A multivariate coefficient or gradient holds the other inputs fixed. Correlated expression, lineage/state proxies, regulatory feedback, selenium availability, medium composition and protein regulation can separate this conditional association from a perturbation phenotype. Do not flip or constrain coefficients solely to match a literature label. A sign can also be stable while replacement importance is zero or negative; the tables retain both measures to avoid confusing output influence with useful prediction.','',
        'The model uses baseline gene-level RNA. Papers often manipulate enzyme activity, protein localization, isoforms, phosphorylation, glycosylation, knockout or inducer dose. No new model was trained and no gene was removed or substituted. In particular, the CTSD/CTSB finding is an evidence annotation issue and does not retroactively change the fitted panel.','',
        'The same overlapping holdouts and repeatedly inspected cohorts are reused. This is an exploratory interpretation audit, not an independent test of gene function. Literature disagreement is descriptive and does not refute a primary study. Literature curation was not blinded to earlier model results, so concordance counts are not a formal validation statistic.','',
        '## Verification and reproduction','',
        'All 120 model hashes and source/input/split hashes were verified against the retained registry and original importance records. Saved coefficients and NN gradients were independently reconstructed on the exact held-out cells; per-repeat and per-lineage results reproduced. Exact Elastic Net zero/sign counts were checked. All 64 genes have explicit literature roles or an uncertainty label and at least one cited source. See [validation](validation.json), [protocol](protocol.json), [evidence](candidate_gene_evidence.json) and [references](references.json).','',
        '```bash','python outputs/erastin_gene_direction_literature/curate_evidence.py',
        'PYTHONPATH="$PWD/work/bulkformer/packages" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache" PYTHONDONTWRITEBYTECODE=1 \\',
        '/home/sji/miniforge3/envs/deeplearning/bin/python outputs/erastin_gene_direction_literature/audit_directions.py','```','',
        '## References','']
    for key,r in refs.items():lines.append(f'{refnum[key]}. [{r["title"]}]({r["url"]})')
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
