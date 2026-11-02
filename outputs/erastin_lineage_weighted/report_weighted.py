"""Render the final data-derived report for the frozen weighting comparison."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path(__file__).resolve().parent
FAMILIES=['ridge','neural_network','xgboost'];SIZES=[40,64,80]
LABELS={'ridge':'Ridge','neural_network':'Neural network','xgboost':'XGBoost'}

def name(f,p,weighted=True):return ('weighted__' if weighted else 'unweighted__')+f+'__'+str(p)
def label(arm):
    w,f,p=arm.split('__');return f'{LABELS[f]}, {p} genes ({w})'

def write_report(summary,paired,effects,wins,counts,validation):
    pooled=summary[summary.scope=='pooled_745'].set_index('model');macro=summary[summary.scope=='macro_19'].set_index('model')
    bp=pooled.rmse_mean.idxmin();bm=macro.rmse_mean.idxmin();br=macro.spearman_mean.idxmax()
    rows=['# Inverse-lineage-frequency training weights for erastin prediction','',
        f'Lowest pooled RMSE across the 18 evaluated arms: **{label(bp)} ({pooled.loc[bp,"rmse_mean"]:.6f})**. '
        f'Lowest equal-lineage mean RMSE: **{label(bm)} ({macro.loc[bm,"rmse_mean"]:.6f})**. '
        f'Highest equal-lineage mean Spearman: **{label(br)} ({macro.loc[br,"spearman_mean"]:.5f})**.','',
        'All nine weighted model/panel combinations completed the same 20 saved leave-lineage-out repeats: '
        '745 cell lines, 734 identity groups, 24 lineage holdouts and five inner tuning folds. '
        'All 4,320 weighted outer models were newly fitted. The nine verified original unweighted prediction arrays were reused for paired comparisons.','',
        '## Results','',
        'Each entry is the mean score across 20 repeats. Scores were computed separately for each repeat; predictions were not averaged across repeats before scoring. '
        'Pooled metrics include all 745 lines. Macro metrics are equal-weight means over the same 19 lineages with n≥10.','',
        '| Model | Genes | Training | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for p in SIZES:
            for weighted in [False,True]:
                arm=name(f,p,weighted);a=pooled.loc[arm];b=macro.loc[arm]
                rows.append(f'| {LABELS[f]} | {p} | {"Weighted" if weighted else "Original"} | {a.spearman_mean:.5f} | {a.rmse_mean:.6f} | {a.r2_mean:.5f} | {b.spearman_mean:.5f} | {b.rmse_mean:.6f} |')
    rows+=['',f'![Weighted comparison]({OUT}/weighted_comparison.png)','','## Matched effect of reweighting','',
        'Positive improvement favors inverse-frequency weighting: Spearman(weighted) − Spearman(original), or RMSE(original) − RMSE(weighted). '
        'Intervals use 5,000 paired whole-lineage bootstrap draws, averaging the per-repeat contrasts within each draw. '
        'They are conditional on the fitted predictions, do not refit models, and are not adjusted for multiple comparisons.','',
        '| Scope | Model | Genes | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---:|---|---:|---:|---|']
    matching={(name(f,p),name(f,p,False)) for f in FAMILIES for p in SIZES}
    for v in paired.itertuples():
        if (v.first_model,v.reference_model) not in matching or v.metric=='r2':continue
        _,f,p=v.first_model.split('__')
        rows.append(f'| {v.scope} | {LABELS[f]} | {p} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','## Consistency across unseen lineages','',
        'Counts below compare mean held-out performance across repeats, among the 19 lineages with n≥10. '
        'A positive count reflects a favorable point estimate, not a separately established effect in that lineage.','',
        '| Model | Genes | Higher lineage Spearman | Lower lineage RMSE |',
        '|---|---:|---:|---:|']
    for v in wins.itertuples():
        if (v.first_model,v.reference_model) not in matching:continue
        _,f,p=v.first_model.split('__');rows.append(f'| {LABELS[f]} | {p} | {v.spearman_favorable_lineages}/19 | {v.rmse_favorable_lineages}/19 |')
    rows+=['',f'![Lineage effects]({OUT}/lineage_weighting_effects.png)','','## Panel expansion under weighted training','',
        '| Model | Panel contrast | Scope | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---|---|---:|---:|---|']
    for v in paired.itertuples():
        aw,af,ap=v.first_model.split('__');bw,bf,bp_=v.reference_model.split('__')
        if aw!='weighted' or bw!='weighted' or af!=bf or v.metric=='r2':continue
        rows.append(f'| {LABELS[af]} | {ap} vs {bp_} | {v.scope} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    audit=pd.read_csv(OUT/'training_subset_weight_audit.tsv',sep='\t');outer=audit[audit.stage=='outer'];inner=audit[audit.stage=='inner']
    weights=pd.read_csv(OUT/'training_lineage_weights.tsv',sep='\t')
    small=weights[weights.stage=='outer'].assign(rare=lambda a:a.n<10).groupby(['repeat','holdout_lineage']).apply(lambda a:a.loc[a.rare,'total_weight'].sum()/a.total_weight.sum(),include_groups=False)
    rows+=['','## Exact weighting and preserved controls','',
        r'For each current fitting subset, $w_i=N/(K n_{\ell(i)})$, where N is its training sample count, K its number of represented lineages, and n its training count for the sample’s lineage. '
        'Weights are recomputed using only that subset. They sum to N, average one, and give each represented lineage total weight N/K. '
        'All represented training lineages participate, including lineages with fewer than 10 samples; the n≥10 restriction applies only to reported macro scoring.','',
        f'- Outer refit per-cell weights range from {outer.weight_min.min():.4f} to {outer.weight_max.max():.4f}; '
        f'K ranges from {outer.represented_lineages.min():.0f} to {outer.represented_lineages.max():.0f}. '
        f'Weight effective sample size, (sum w)²/sum(w²), ranges from {outer.effective_sample_size.min():.1f} to {outer.effective_sample_size.max():.1f}; this is a weight-concentration diagnostic, not a biological sample count.',
        f'- Inner-fit per-cell weights range from {inner.weight_min.min():.4f} to {inner.weight_max.max():.4f}; '
        f'K ranges from {inner.represented_lineages.min():.0f} to {inner.represented_lineages.max():.0f}.',
        f'- Lineages represented by fewer than 10 training samples receive {100*small.min():.2f}–{100*small.max():.2f}% of total outer training weight. The macro-19 score excludes lineages with fewer than 10 total cohort samples, so it is not the identical population balance as the all-lineage training objective.',
        '- X/y standardization remains unweighted and uses only current training rows, matching the original comparison. Ridge uses weighted centering for its unpenalized intercept.',
        '- Hyperparameter selection remains the minimum unweighted equal-mean five-fold inner RMSE. This experiment changes training loss weights while retaining the original tuning objective.',
        '- Ridge retains the original 17-alpha grid. Neural network retains the four width/penalty candidates (16/32 tanh units, alpha 10/100), L-BFGS, and the same three initializations per repeat. XGBoost retains the original 36 candidates, CPU hist, 0.03 learning rate, 0.8 row/column sampling, and per-repeat seeds.',
        '- Weights multiply the squared-error training terms in all families. Normalization to mean one retains the original regularization scale. XGBoost retains uniform row sampling at 0.8; equality is in full-set weight totals, not each stochastic subsample. Its minimum-child-weight grid now operates on weighted Hessian mass.',
        '- Held-out lineage membership and related identity groups are excluded from both inner fitting and outer refitting. No lineage label is supplied as an input feature. All repeats and lineages are retained.','',
        '| Training lineage | Outer-refit per-cell weight range | Training cell-count range when present |',
        '|---|---:|---:|']
    grouped=weights[(weights.stage=='outer')&(weights.repeat==1)].groupby('lineage').agg(wmin=('weight_per_cell','min'),wmax=('weight_per_cell','max'),nmin=('n','min'),nmax=('n','max')).sort_values('wmax',ascending=False)
    for lineage,v in grouped.iterrows():rows.append(f'| {lineage} | {v.wmin:.3f}–{v.wmax:.3f} | {int(v.nmin)}–{int(v.nmax)} |')
    rows+=['','## Validation and reproducibility','',
        f'- All {validation["new_outer_models"]:,} weighted outer models reconstructed their saved predictions; maximum absolute discrepancy {validation["prediction_max_error"]:.3g}.',
        f'- All {validation["reconstructed_candidate_fold_scores"]:,} candidate-fold scores were recomputed; maximum RMSE discrepancy {validation["inner_rmse_max_error"]:.3g}. Selected candidates match the original unweighted inner-scoring rule.',
        f'- All {validation["training_subset_weight_checks"]:,} fitting-subset weight summaries were reconstructed from their training memberships; equal total lineage weights and training-only scaling were checked.',
        '- Preflight: neural-network unit weights exactly reproduced the original implementation on synthetic data and all three actual panels for Lung/Thyroid in repeat 1. Weighted neural gradients matched independent finite differences at three parameter points; direct weighted-objective calculations also matched. Weighted ridge and XGBoost agreed with separate sklearn API checks.',
        '- Selected settings were refitted for all nine weighted arms in Lung/Thyroid for repeats 1/20, including the five corresponding inner fits. Ridge and XGBoost use separate sklearn APIs for these checks; neural refits use the verified weighted adapter and check deterministic replay.',
        f'- Training warnings: {validation["warning_count"]:,}; convergence warnings: {validation["convergence_warning_count"]:,}. Warning summaries and per-fit diagnostics are saved.',
        '- Original prediction scores, source hashes, exact saved split arrays, sample order, targets and initialization seeds were verified. Full weighted models, inner predictions, tuning tables, weight audits, bootstrap draws and source scripts are included in the archive.','',
        '## Interpretation limits','',
        'These results concern transfer to held-out cancer lineages in this reused cell-line cohort. The 20 repeats perturb inner tuning and initialization, not the biological sample; consistency across seeds does not supply 20 independent biological validations. '
        'Whole-lineage bootstrap intervals reflect uncertainty across the observed held-out lineages, conditional on the fitted models. '
        'Extremely small training lineages receive large per-cell weights and can substantially affect fitting. Equal training weight does not guarantee equal accuracy, and no observed effect establishes causal ferroptosis biology. '
        'Inner folds mix training lineages, as in the original protocol; only outer tests measure unseen-lineage transfer. '
        'The original harmonized erastin AUC provenance limitation remains unresolved, and no independent drug-response cohort was used.','',
        '## Files','',
        '- `summary.tsv`: all 18 arms, pooled and macro scores with seed dispersion.',
        '- `paired_comparisons.tsv` and `paired_seed_differences.tsv`: matching reweighting contrasts plus weighted panel/family comparisons.',
        '- `lineage_paired_effects.tsv` and `per_lineage_metrics.tsv`: held-out lineage effects.',
        '- `training_subset_weight_audit.tsv` and `training_lineage_weights.tsv`: exact training-weight totals, counts and concentration.',
        '- `selected_parameters.tsv`, `tuning_scores.tsv`, `parameter_counts.tsv`, `validation.json`, `preflight_validation.json`, `protocol.json`: controls and audits.',
        '- `predictions.npz`, `bootstrap_draws.npz`, `folds/`: predictions, uncertainty draws and all fitted weighted models.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
