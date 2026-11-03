# Retained 64-gene models

On **2026-11-02**, the user selected the original 64-gene ferroptosis models for
further work: ridge, Elastic Net and the neural network, with unweighted training
and no culture covariates or added EMT/DNA-repair genes.

The user reaffirmed this decision after the gene-importance analysis and the
nested top-10/top-20 NN benchmark. The full 64-gene NN had the lowest mean test
RMSE in both CTRP and PRISM across the 20 shared splits. The reduced panels did
not establish a prediction-error improvement. The canonical panel and all
120 retained model references remain unchanged.

`model_registry.json` is the canonical index. It records the ordered panel,
cohort-specific input/split/protocol checksums, model settings, summary accuracy,
and the original paths and checksums for all **120 saved outer models**:
three families × 20 repeats × two response cohorts (CTRP and PRISM).

The model binaries remain at their original paths. The registry does not copy,
delete or retrain them. These are repeated-holdout models, not an all-cohort final
refit. The assays remain separate.

Earlier comparisons are preserved as research history. Retaining the baseline
does not overwrite their reports or imply that expansion was proven equivalent.

Rebuild and verify this registry from the repository root:

```bash
python scripts/register_retained_models.py
```
