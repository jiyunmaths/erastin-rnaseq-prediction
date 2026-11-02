"""Publication-style figures and a self-contained culture-covariate report."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
OUT=Path(__file__).resolve().parent
FAMILIES=['ridge','elastic_net','neural_network'];LABELS=dict(ridge='Ridge',elastic_net='Elastic Net',neural_network='Neural network');COLORS=['#3274a1','#e1812c'];name=lambda f,p,v:f+'__'+str(p)+'__'+v

def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)

def render(summary,paired,effects,wins,validation):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'});s=summary.set_index('model');primary=paired[(paired.metric=='pooled_rmse')&(paired.reference_model.str.endswith('__expression'))].copy();primary=primary.set_index('first_model');repeat=pd.read_csv(OUT/'per_repeat_pooled_metrics.tsv',sep='\t');prep=pd.read_csv(OUT/'outer_preprocessing.tsv',sep='\t');params=pd.read_csv(OUT/'selected_parameters.tsv',sep='\t');protocol=json.load(open(OUT/'protocol.json'));cov=json.load(open(OUT/'metadata_coverage.json'))
    # Match the panels and families, with culture-only controls shown separately.
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.4),sharey=True,layout='constrained')
    for ax,f in zip(axes,FAMILIES):
        for j,v in enumerate(['expression','culture']):
            xx=np.array([0,1])+(j-.5)*.28;yy=[s.loc[name(f,p,v),'pooled_rmse_mean'] for p in [5,64]];ax.bar(xx,yy,.26,color=COLORS[j],label='Genes only' if j==0 else 'Genes + culture',zorder=2)
            for x,p in zip(xx,[5,64]):
                vals=repeat[repeat.model==name(f,p,v)].sort_values('repeat').rmse.to_numpy();ax.scatter(x+np.linspace(-.07,.07,20),vals,s=10,color='black',alpha=.32,zorder=3)
        ax.axhline(s.loc[name(f,0,'culture'),'pooled_rmse_mean'],color='#28855b',ls='--',label='Culture only',lw=1.5);ax.axhline(s.loc['training_mean','pooled_rmse_mean'],color='.45',ls=':',label='Training mean',lw=1.5);ax.set_xticks([0,1],['5 genes','64 genes']);ax.set_title(LABELS[f]);ax.set_ylim(0,max(.18,float(repeat.rmse.max())*1.03));ax.grid(axis='y',alpha=.16,zorder=0)
    axes[0].set_ylabel('Held-out CTRP RMSE (lower is better)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False);fig.suptitle('Culture metadata added to gene expression — same 20 test splits',fontsize=13);save(fig,'model_comparison')
    order=[name(f,p,'culture') for f in FAMILIES for p in [5,64]];ys=np.arange(6)[::-1];fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for yy,arm in zip(ys,order):
        v=primary.loc[arm];ax.plot([v.conditional_ci_low,v.conditional_ci_high],[yy,yy],lw=2,color='#3274a1');ax.plot(v.improvement,yy,'o',color='#3274a1');ax.text(1.02,yy,f'{int(v.favorable_repeats)}/20 splits',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.axvline(0,color='.4',lw=1);ax.set_yticks(ys,[LABELS[a.split('__')[0]]+' · '+a.split('__')[1]+' genes' for a in order]);ax.set_xlabel('RMSE reduction from adding culture (positive favors culture)');ax.set_title('Mean paired change and conditional 95% lineage-bootstrap interval');ax.grid(axis='x',alpha=.15);save(fig,'paired_culture_effects')
    labels=sorted(effects.lineage.unique());matrix=np.array([[effects[(effects.first_model==a)&(effects.reference_model==a.replace('__culture','__expression'))&(effects.lineage==l)].rmse_improvement.iloc[0] for a in order] for l in labels]);bound=float(np.max(np.abs(matrix)));fig,ax=plt.subplots(figsize=(10.8,9),layout='constrained');im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_yticks(range(23),labels);ax.set_xticks(range(6),[LABELS[a.split('__')[0]]+'\n'+a.split('__')[1]+' genes' for a in order]);ax.set_title('Culture effects vary by lineage\nMean held-out RMSE reduction across the same 20 splits')
    for i in range(23):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>.55*bound else 'black')
    fig.colorbar(im,ax=ax,label='RMSE reduction (positive favors culture)',shrink=.75);save(fig,'lineage_culture_effects')
    best=summary[summary.family!='training_mean'].sort_values('pooled_rmse_mean').iloc[0];favorable=int((primary.improvement>0).sum());clear=int((primary.conditional_ci_low>0).sum());harm=int((primary.conditional_ci_high<0).sum());best_label=LABELS[best.family]+(' culture only' if best.panel==0 else f', {int(best.panel)} genes'+(' + culture' if best['view']=='culture' else ' only'))
    rows=['# Adding DepMap culture metadata to CTRP erastin prediction','',f'**Completed 180 new outer fits and reused the 120 unchanged gene-only models.** All 300 models passed the checks below. Culture augmentation lowered mean pooled test RMSE in {favorable}/6 matched gene-panel/model comparisons. Of the six conditional 95% intervals, {clear} lie above zero (benefit), {harm} below zero (harm), and {6-clear-harm} include zero. These are exploratory, unadjusted comparisons.','',f'Lowest mean pooled RMSE among all 15 fitted arms: **{best_label}**, {best.pooled_rmse_mean:.6f}. Ranking the observed means is descriptive and does not establish superiority.','',
        '**The culture annotations describe DepMap onboarding. They are not verified CTRP assay conditions.** Any predictive association may reflect lineage, source, or other cell-line properties; it cannot establish a causal growth-medium effect.','',
        '## What was added','',
        '| Covariate | Coverage / encoding |','|---|---|',
        '| GrowthPattern | 548 adherent, 45 mixed, 148 suspension, 3 unknown. Unknown is explicit missing. |',
        '| Medium identity | 704/744 annotated; 87 distinct MF codes. Training categories with fewer than 5 rows are pooled as OTHER; unseen codes use OTHER. |',
        '| Basal-medium family | Recipe prefix before first +; same training-only rare-category grouping. |',
        '| Recorded serum percentage | Explicit % FBS in 703 recipes; training-median imputation for 41 unknown percentages (40 absent media plus one serum-free recipe with no numeric FBS %), plus a missingness indicator. No zero inferred. |','',
        'The same PRISM covariate set is retained. PlateCoating is the literal None for all 744 lines; ModelDerivationMaterial and CulturedResistanceDrug are empty. ModelTreatment has one EBV record and is outside the fixed covariate set. SerumFreeMedia has 703 False, one True and 40 missing; no extra serum-free flag is added. The True recipe lacks explicit FBS % and remains missing numerically. No basal nutrient concentrations or unrecorded supplements were inferred. Exact full recipe is not encoded separately from its one-to-one medium code.','',
        'The downloaded Model.csv has an important schema inconsistency: **OnboardedMedia contains MF codes; FormulationID contains recipe text**, while the accompanying README descriptions imply the opposite. Code–recipe mapping is one-to-one in the retained cohort. Raw column names and values are preserved in culture_metadata.tsv and the observed mapping is saved in medium_dictionary.tsv. Source files match their original recorded SHA256 hashes.','',
        'Source: [DepMap 24Q4 Public v1](https://doi.org/10.25452/figshare.plus.27993248.v1). Outcome: unchanged harmonized CTRP erastin AUC from the [recorded redistribution archive](https://doi.org/10.5281/zenodo.22125121). Archive checksums were verified in the original preparation; the exact upstream AUC generation recipe remains unverified. The local source README and source manifests are the provenance for the field audit.','',
        '## Gene-only versus gene-plus-culture test scores','',
        '| Family | Genes | RMSE genes only | RMSE + culture | Spearman genes only | Spearman + culture | R² genes only | R² + culture |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for p in [5,64]:
            a=s.loc[name(f,p,'expression')];b=s.loc[name(f,p,'culture')];rows.append(f'| {LABELS[f]} | {p} | {a.pooled_rmse_mean:.6f} | {b.pooled_rmse_mean:.6f} | {a.pooled_spearman_mean:.4f} | {b.pooled_spearman_mean:.4f} | {a.pooled_r2_mean:.4f} | {b.pooled_r2_mean:.4f} |')
    rows+=['','Metrics are computed separately on each 149-line test set, then averaged over 20 repeats; predictions are not stacked as independent observations. Higher negative-AUC predictions mean greater erastin sensitivity. The original harmonized CTRP target array is retained without transformation beyond its existing sign convention.','',f'![Model comparison]({OUT}/model_comparison.png)','','## Paired augmentation effects','',
        '| Family | Genes | Mean RMSE reduction | Relative reduction | Favorable splits | Favorable lineages | Conditional 95% interval |','|---|---:|---:|---:|---:|---:|---|']
    for f in FAMILIES:
        for p in [5,64]:
            a=name(f,p,'culture');b=name(f,p,'expression');v=primary.loc[a];w=wins[(wins.first_model==a)&(wins.reference_model==b)].iloc[0];rows.append(f'| {LABELS[f]} | {p} | {v.improvement:+.6f} | {100*v.improvement/s.loc[b,"pooled_rmse_mean"]:+.3f}% | {int(v.favorable_repeats)}/20 | {int(w.rmse_favorable_lineages)}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Positive reduction favors adding culture. Favorable lineages count a positive mean within-lineage RMSE reduction; small lineages can have only one held-out line per repeat, so these counts are descriptive. An interval containing zero does not establish equivalence.','',f'![Paired effects]({OUT}/paired_culture_effects.png)','','| Family | Genes | Macro-23 RMSE genes only | Macro-23 RMSE + culture | Mean macro-23 reduction | Conditional 95% interval |','|---|---:|---:|---:|---:|---|']
    for f in FAMILIES:
        for p in [5,64]:
            a=name(f,p,'culture');b=name(f,p,'expression');v=paired[(paired.first_model==a)&(paired.reference_model==b)&(paired.metric=='macro_all_rmse')].iloc[0];rows.append(f'| {LABELS[f]} | {p} | {s.loc[b,"macro_all_rmse_mean"]:.6f} | {s.loc[a,"macro_all_rmse_mean"]:.6f} | {v.improvement:+.6f} | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Macro-23 gives equal weight to the 23 lineage-specific test RMSEs. Macro-19, MAE, Pearson, R², full repeat values and all paired intervals are also saved.','',f'![Lineage effects]({OUT}/lineage_culture_effects.png)','','## Culture-only controls','',
        '| Family | RMSE | Spearman | R² | Macro-23 RMSE |','|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        v=s.loc[name(f,0,'culture')];rho=f'{v.pooled_spearman_mean:.4f}' if np.isfinite(v.pooled_spearman_mean) else 'NA';rows.append(f'| {LABELS[f]} | {v.pooled_rmse_mean:.6f} | {rho} | {v.pooled_r2_mean:.4f} | {v.macro_all_rmse_mean:.6f} |')
    v=s.loc['training_mean'];rows.append(f'| Outer-training mean | {v.pooled_rmse_mean:.6f} | NA | {v.pooled_r2_mean:.4f} | {v.macro_all_rmse_mean:.6f} |')
    rows+=['','## Five versus 64 genes with culture included','',
        '| Family | RMSE reduction: 64 vs 5 | Relative reduction | Favorable splits for 64 | Conditional 95% interval |','|---|---:|---:|---:|---|']
    for f in FAMILIES:
        first=name(f,5,'culture');ref=name(f,64,'culture');v=paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric=='pooled_rmse')].iloc[0]
        rows.append(f'| {LABELS[f]} | {-v.improvement:+.6f} | {-100*v.improvement/s.loc[first,"pooled_rmse_mean"]:+.3f}% | {20-int(v.favorable_repeats)}/20 | [{-v.conditional_ci_high:+.6f}, {-v.conditional_ci_low:+.6f}] |')
    rows+=['','All three 64-gene augmented models have lower mean RMSE than their five-gene counterparts. Only the ridge panel interval excludes zero, narrowly, under these conditional unadjusted intervals. This is a previously specified panel contrast; positive reductions in this table favor 64 genes.','']
    rows+=['','All culture-only models use the same preprocessing, tuning grids and inner/outer splits as augmented arms. Constant predictors retain undefined correlations as NA; defined-repeat counts are saved. The training-mean prediction uses only current outer-training outcomes.','',
        '## Unchanged split design and tuning','',
        '- 744 cell lines, 733 identity groups, 23 lineages. Cervix remains excluded as in the baseline. Exact prior shared 20 outer masks and five inner folds per repeat; unweighted training. Each outer split has 595 training and 149 testing lines, with every lineage on both sides and no shared identity group. No new cohort exclusion.',
        '- Every inner training fold contains all 23 lineages; validation contains 22. The single outer-training Ampulla line remains in every inner training fit and is never used for inner validation. Categories, rare-category pooling, serum imputation and all scaling are fitted inside the current inner training fold, then refitted on outer training for the selected model. Lineage and cell identity are not predictor columns.',
        '- Standardize expression and serum percentage only. One-hot categories and missingness indicators are centered without variance standardization. Reserved missing and OTHER columns distinguish absent annotations from rare/unseen categories. Column means from training center the final joint design.',
        '- Ridge: 17 alphas, 1e-4 to 1e4. Elastic Net retains the same 41 alphas, 10 to 1e-4, crossed with L1 ratios .01/.05/.1/.25/.5/.75/.9/1. Mixed-feature fits use tighter tol 1e-11 and max_iter 200,000; the reused baselines used tol 1e-9 and max_iter 50,000. This is a numerical precision change, not a new tuning grid.',
        '- Overlapping medium indicators make pure-Lasso coefficients potentially nonunique. Inner candidates at L1 ratio 1 use native cold starts to match outer refits. Other mixing ratios use the original warm-start path algorithm. Native prediction parity is checked; augmented-row Lasso equivalence is assessed by objective and optimality residual for pure Lasso because unseen-combination predictions need not be unique across optimal coefficient vectors.',
        '- Neural network: 16/32 tanh units, alpha 10/100; three initializations averaged using the same saved triplets. L-BFGS, max_iter 2,000, max_fun 40,000, tol .001, no early stopping. Native unweighted MLP parity with the previous unit-weight adapter passed the independent preflight.',
        '- Select by equal-mean five-fold RMSE in original target units, with first grid member breaking ties. The 5-gene panel remains SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. The 64-gene panel and expression values are unchanged.',
        '- Culture adds columns without changing the candidate grids. Penalization and selected architecture can still change when features are added; this is an operational prediction comparison, not an isolated causal culture-effect estimate.','',
        '| Input genes | Outer joint-feature count range |','|---:|---:|']
    for p in [0,5,64]:
        d=prep[prep.model.str.contains('__'+str(p)+'__')];rows.append(f'| {p} | {int(d.features.min())}–{int(d.features.max())} |')
    rows+=['','## Validation and interpretation limits','',
        f'- All 300 outer models audited: 180 newly fitted, 120 baseline model files reused byte-for-byte. Original baseline test predictions and mean scores reproduced. Verified all source/input/script hashes and split masks.',
        f'- Replayed all 40 selected gene-only Elastic Net settings at the tighter numerical precision; maximum held-out prediction change {validation["baseline_elastic_net_precision_max_prediction_change"]:.3g}. This checks selected-fit precision without changing saved baseline tuning or predictions.',
        f'- Reconstructed {validation["candidate_fold_scores_checked"]:,} candidate-fold scores, {validation["identity_exclusion_checks"]:,} identity exclusions and {validation["training_only_preprocessor_checks"]:,} training-only design reconstructions. Maximum saved-prediction error {validation["prediction_max_error"]:.3g}; inner-RMSE error {validation["inner_rmse_max_error"]:.3g}.',
        f'- Recomputed all 300 selected NN-member objectives (maximum error {validation["neural_objective_max_error"]:.3g}); checked 100 selected outer Elastic Net objectives and recorded residuals for 164,000 inner candidate solutions. Maximum Elastic Net optimality residual {validation["elastic_net_max_kkt_residual"]:.3g}.',
        '- Independently reconstructed mixed-feature matrices using raw values and explicit indicators. Verified unseen/missing category handling, training-only medians/scalers/category thresholds, native prediction parity, augmented-Lasso objective/optimality equivalence for pure Lasso and prediction equivalence for other mixing ratios, and the previous unit-weight NN adapter. Independent selected-setting inner/outer refits passed for every new arm in repeats 1 and 20.',
        f'- Convergence warnings: {validation["convergence_warning_count"]} across candidate/outer fits; selected outer convergence warnings: {validation["selected_outer_convergence_warning_count"]}. Other warning categories are saved.',
        '- Intervals use 5,000 paired whole-lineage bootstrap draws shared across arms/repeats. They condition on fitted models and the observed cohort/splits; no model refitting, multiplicity correction or post-selection correction. Independent expanded-sample SciPy checks passed, including undefined constant-prediction correlations.',
        '- This is an exploratory internal CTRP comparison informed by earlier results. The five-gene panel was chosen after earlier held-out importance analysis on these same splits; tuning inside training does not remove this earlier test-informed selection. Overlapping 20 holdouts are not independent biological replications; there is no untouched final test cohort. Evaluation is for new cell lines from represented lineages. It does not establish unseen-lineage, unseen-medium, patient or clinical generalization.',
        '- Onboarding media are proxies for the requested growth conditions. Missingness, rare recipes and highly imbalanced growth patterns limit what can be learned. This analysis does not identify a causal medium effect or show that changing a cell line’s culture recipe changes erastin response.','',
        '## Files','',
        '- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv: overall accuracy and paired effects.',
        '- per_lineage_metrics.tsv, lineage_paired_effects.tsv, lineage_win_counts.tsv: lineage-specific accuracy and effect consistency.',
        '- culture_metadata.tsv, medium_dictionary.tsv, metadata_coverage.json, culture_coverage_by_lineage.tsv: raw/derived onboarding fields and overall/lineage coverage audit.',
        '- panel_inputs.npz, shared_splits.npz: unchanged RNA, outcomes, identities and train/test/validation memberships.',
        '- folds/, selected_parameters.tsv, tuning_scores.tsv, outer_preprocessing.tsv: saved fitted models, predictions and tuning/preprocessing diagnostics. Use the saved preprocessor before new culture-model predictions.',
        '- linear_coefficients.tsv: fitted coefficients in the model design (not causal importance; continuous and binary columns have different scaling conventions).',
        '- protocol.json, source_manifest.json, preflight_validation.json, validation.json, visual_qa.json, artifact_manifest.json and scripts: reproducibility and integrity evidence.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
