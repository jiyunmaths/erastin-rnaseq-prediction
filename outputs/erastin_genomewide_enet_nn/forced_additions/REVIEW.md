# Forced-addition neural-network diagnostic

This secondary diagnostic was registered after early training-validation choices favored the core-only model, before inspection of the new NN held-out scores. It uses the best nonempty setting from the unchanged inner NN grid. All original 64 genes remain; no new inner fitting, tuning parameters or test-driven gene choices are added. The primary zero-fallback comparison remains unchanged.

| Assay | Core64 NN RMSE | Forced additions NN RMSE | RMSE reduction | Conditional 95% interval | Favorable repeats /20 |
|---|---:|---:|---:|---:|---:|
| CTRP | 0.116968 | 0.124985 | -0.008018 | [-0.011683, -0.004321] | 1 |
| PRISM | 0.138203 | 0.146790 | -0.008587 | [-0.011720, -0.004970] | 1 |

Positive RMSE reductions favor additions. Intervals are exploratory paired whole-lineage bootstrap intervals conditional on fits; overlapping seeds are not independent experiments. Models, predictions, selected parameters, lineage effects and validation are saved in this directory.
