#!/usr/bin/env python3
"""Rank all 64 retained NN inputs without fitting or selecting another model.

Run with the pinned sklearn 1.5.2 analysis environment. The earlier importance
calculation is the source of exhaustive same-lineage training-donor replacement;
this script independently checks its aggregation, gradients, intervals and exact
linkage to the current retained-model registry before exporting readable ranks.
"""
import hashlib
import importlib.metadata
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
SOURCE = WS / "outputs/erastin_prism_ctrp_gene_importance"
REGISTRY = WS / "outputs/retained_64_gene_models/model_registry.json"
sys.path.insert(0, str(SOURCE))  # Original saved WeightedMLPRegressor class.
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv(name, frame):
    frame.to_csv(OUT / name, index=False, float_format="%.12g")


def predict(model, z):
    standardized = np.mean([v.predict(z) for v in model["members"]], axis=0)
    return model["y_scaler"].inverse_transform(standardized[:, None]).ravel()


def gradient(model, z):
    grad = np.zeros_like(z)
    for member in model["members"]:
        assert member.activation == "tanh" and member.out_activation_ == "identity"
        assert len(member.coefs_) == 2
        h = np.tanh(z @ member.coefs_[0] + member.intercepts_[0])
        grad += ((1 - h*h) * member.coefs_[1].ravel()) @ member.coefs_[0].T
    return grad * model["y_scaler"].scale_[0] / len(model["members"])


def main():
    OUT.mkdir(exist_ok=True)
    records = {}

    def record(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        records[str(path.relative_to(WS))] = dict(
            path=str(path.relative_to(WS)), sha256=digest, bytes=path.stat().st_size)

    record(REGISTRY)
    record(SOURCE / "artifact_manifest.json")
    known = {v["path"]: v["sha256"] for v in json.loads(
        (SOURCE / "artifact_manifest.json").read_text())["files"]}

    def verified(name):
        record(SOURCE / name, known[name])
        return SOURCE / name

    registry = json.loads(REGISTRY.read_text())
    assert registry["gene_count"] == 64 and not registry["culture_covariates"]
    original_sources = {v["path"]: v["sha256"] for v in json.loads(
        verified("source_manifest.json").read_text())["files"]}
    previous_validation = json.loads(verified("validation.json").read_text())
    assert previous_validation["passed"] and previous_validation["no_refitting"]
    source_summary = pd.read_csv(verified("gene_importance_summary.tsv"), sep="\t")
    source_lineage = pd.read_csv(verified("per_lineage_importance.tsv"), sep="\t")
    source_repeat = pd.read_csv(verified("per_repeat_importance.tsv"), sep="\t")
    boot = np.load(verified("bootstrap_draws.npz"))
    for name in ["protocol.json", "calculate_importance.py", "review_importance.py",
                 "weighted_mlp.py", "coefficient_sensitivity_by_repeat.tsv"]:
        verified(name)

    all_rows, split_rows, lineage_rows, checks, stability_rows = [], [], [], [], []
    maxima = {k: 0.0 for k in ["prediction", "gradient", "finite_difference", "aggregation",
                               "relative_interval", "direct_replacement"]}
    for ds, cohort in registry["cohorts"].items():
        stem = ds.lower()
        inp = np.load(verified(stem + "_inputs.npz"))
        arr = np.load(verified(stem + "_importance_arrays.npz"))
        genes, x, y = inp["genes"], inp["x"], inp["y"]
        np.testing.assert_array_equal(genes, registry["genes_in_input_order"])
        assert len(set(genes)) == 64
        for entry in cohort["input_and_validation_files"].values():
            record(WS / entry["path"], entry["sha256"])
            if entry["path"] in original_sources:
                assert original_sources[entry["path"]] == entry["sha256"]
        native = np.load(WS / cohort["input_and_validation_files"]["panel_inputs.npz"]["path"])
        splits = np.load(WS / cohort["input_and_validation_files"]["shared_splits.npz"]["path"])
        for akey, nkey in [("x", "x_64"), ("y", "y"), ("ids", "sample_ids"),
                          ("groups", "group_ids"), ("lineage", "lineage"), ("genes", "genes_64")]:
            np.testing.assert_array_equal(inp[akey], native[nkey])
        np.testing.assert_array_equal(inp["test"], splits["outer_test"])
        d = source_summary.query(
            "dataset == @ds and model == 'neural_network' and scope == 'native_pooled'"
        ).set_index("gene").loc[genes].copy()
        n = arr["test_n"]
        repeat_effect = np.einsum("rlg,l->rg", arr["D"][2], n) / n.sum()
        point = repeat_effect.mean(axis=0)
        base = np.mean(arr["B"][2] @ n / n.sum())
        np.testing.assert_allclose(point, d.delta_mse, rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(base, d.baseline_mse, rtol=1e-10)
        np.testing.assert_array_equal(rankdata(-point), d["rank"])
        maxima["aggregation"] = max(maxima["aggregation"], float(np.max(np.abs(point-d.delta_mse))))
        ci = np.percentile(100 * boot[f"{ds}__neural_network__native_pooled__delta_mse"] /
                           boot[f"{ds}__neural_network__native_pooled__baseline_mse"][:, None],
                           [2.5, 97.5], axis=0)
        np.testing.assert_allclose(ci, d[["relative_ci_low", "relative_ci_high"]].to_numpy().T,
                                   rtol=1e-9, atol=1e-10)
        maxima["relative_interval"] = max(maxima["relative_interval"], float(np.max(
            np.abs(ci-d[["relative_ci_low", "relative_ci_high"]].to_numpy().T))))
        gradients, per_seed_gradients = [], []
        for entry in cohort["models"]["neural_network"]["outer_models"]:
            r = entry["repeat"] - 1
            for file in entry["files"].values():
                assert original_sources[file["path"]] == file["sha256"]
                record(WS / file["path"], file["sha256"])
            model = joblib.load(WS / entry["files"]["joblib"]["path"])
            fit = np.load(WS / entry["files"]["npz"]["path"])
            te, tr = fit["test_rows"], fit["train_rows"]
            np.testing.assert_array_equal(te, np.flatnonzero(inp["test"][r]))
            np.testing.assert_array_equal(tr, np.flatnonzero(~inp["test"][r]))
            assert set(inp["groups"][tr]).isdisjoint(inp["groups"][te])
            assert set(inp["lineage"][tr]) == set(inp["lineage"][te]) == set(inp["labels"])
            z = model["x_scaler"].transform(x[te])
            pred, grad = predict(model, z), gradient(model, z)
            np.testing.assert_allclose(pred, fit["test_prediction"], rtol=0, atol=1e-12)
            repeat = np.load(verified(f"repeats/{stem}__r{r+1:02d}.npz"))
            verified(f"repeats/{stem}__r{r+1:02d}.json")
            np.testing.assert_allclose(grad, repeat["standardized_gradient"][2], rtol=0, atol=1e-12)
            np.testing.assert_allclose(repeat_effect[r], repeat["train_reference_delta_mse"][2].mean(axis=0),
                                       rtol=0, atol=1e-12)
            maxima["prediction"] = max(maxima["prediction"], float(np.max(np.abs(pred-fit["test_prediction"]))));
            maxima["gradient"] = max(maxima["gradient"], float(np.max(np.abs(grad-repeat["standardized_gradient"][2]))))
            gradients.append(grad)
            per_seed_gradients.append(grad.mean(axis=0))
            if r in [0, 19]:
                eps = 1e-5
                zz = np.repeat(z[:1], 64, axis=0)
                plus, minus = zz + np.eye(64)*eps, zz - np.eye(64)*eps
                finite = (predict(model, plus)-predict(model, minus))/(2*eps)
                err = float(np.max(np.abs(finite-grad[0])))
                assert err < 1e-8
                maxima["finite_difference"] = max(maxima["finite_difference"], err)
                # Direct exhaustive replacement check for every gene, one receiver.
                donors = tr[inp["lineage"][tr] == inp["lineage"][te[0]]]
                assert set(inp["groups"][donors]).isdisjoint(inp["groups"][te])
                direct = []
                for j in range(64):
                    xx = np.repeat(x[te[:1]], len(donors), axis=0)
                    xx[:, j] = x[donors, j]
                    pp = predict(model, model["x_scaler"].transform(xx))
                    direct.append(np.mean((pp-y[te[0]])**2)-(pred[0]-y[te[0]])**2)
                err = float(np.max(np.abs(np.array(direct)-repeat["train_reference_delta_mse"][2, 0])))
                assert err < 1e-11
                maxima["direct_replacement"] = max(maxima["direct_replacement"], err)
            checks.append(dict(dataset=ds, repeat=r+1, members=len(model["members"]),
                               train=len(tr), test=len(te), saved_model_verified=True))
        grad = np.concatenate(gradients)
        signed, rms = grad.mean(axis=0), np.sqrt(np.mean(grad**2, axis=0))
        for actual, col in [(signed, "mean_standardized_slope"), (rms, "rms_standardized_slope"),
                            (np.abs(grad).mean(axis=0), "mean_absolute_standardized_slope"),
                            ((grad>0).mean(axis=0), "positive_gradient_fraction")]:
            np.testing.assert_allclose(actual, d[col], rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(rankdata(-rms), d["sensitivity_rank"])
        ranks_by_seed = rankdata(-repeat_effect, axis=1)
        np.testing.assert_array_equal((repeat_effect>0).sum(axis=0), d.positive_repeats)
        np.testing.assert_array_equal(((ranks_by_seed<=10)&(repeat_effect>0)).sum(axis=0), d.top10_repeats)
        rref = source_repeat.query("dataset == @ds and model == 'neural_network'")
        for r in range(20):
            ref = rref[rref["repeat"] == r+1].set_index("gene").loc[genes]
            np.testing.assert_array_equal(ranks_by_seed[r], ref["rank"])
            for j, gene in enumerate(genes):
                split_rows.append(dict(dataset=ds, repeat=r+1, gene=gene,
                    predictive_rank=int(ranks_by_seed[r, j]), delta_mse=repeat_effect[r, j],
                    signed_standardized_slope=per_seed_gradients[r][j]))
        d["input_order"] = np.arange(1, 65)
        d["higher_expression_predicts"] = np.where(signed>0, "higher sensitivity", "lower sensitivity")
        d["gradient_direction_agreement_fraction"] = np.where(signed>0, (grad>0).mean(axis=0), (grad<0).mean(axis=0))
        per_seed_gradients = np.array(per_seed_gradients)
        d["splits_with_matching_mean_gradient_direction"] = (
            np.sign(per_seed_gradients)==np.sign(signed)).sum(axis=0)
        rename = dict(rank="predictive_rank", relative_mse_increase_percent="relative_mse_increase_pct",
                      relative_ci_low="relative_ci_low_pct", relative_ci_high="relative_ci_high_pct",
                      positive_repeats="positive_splits", top10_repeats="positive_top10_splits",
                      sensitivity_rank="gradient_rank", positive_native_large_lineages="positive_large_lineages",
                      native_large_lineages="eligible_large_lineages", positive_common13="positive_common13_lineages")
        d = d.reset_index().rename(columns=rename)
        int_cols = ["predictive_rank", "gradient_rank", "positive_splits", "positive_top10_splits",
                    "positive_large_lineages", "eligible_large_lineages", "positive_common13_lineages",
                    "rank_min", "rank_max", "positive_native_lineages", "positive_strict7", "top10_common13"]
        d[int_cols] = d[int_cols].astype(int)
        d = d.sort_values("predictive_rank")
        first = ["predictive_rank", "gene", "relative_mse_increase_pct", "relative_ci_low_pct",
                 "relative_ci_high_pct", "higher_expression_predicts", "positive_splits", "positive_top10_splits",
                 "rank_median", "rank_min", "rank_max", "positive_large_lineages", "eligible_large_lineages",
                 "gradient_rank", "mean_standardized_slope", "rms_standardized_slope",
                 "gradient_direction_agreement_fraction", "splits_with_matching_mean_gradient_direction"]
        d = d[first + [c for c in d if c not in first]]
        csv(stem+"_64_gene_ranking.csv", d)
        csv(stem+"_64_gene_gradient_ranking.csv", d.sort_values("gradient_rank"))
        all_rows.append(d)
        lin = source_lineage.query("dataset == @ds and model == 'neural_network'")
        lineage_rows.append(lin)
        # Recompute descriptive pairwise rank stability (no independence claims).
        for scope, values in [("between_splits", repeat_effect),
                              ("between_large_lineages", arr["D"][2].mean(axis=0)[arr["cohort_n"]>=10])]:
            cor = [float(spearmanr(a, b).statistic) for a,b in itertools.combinations(values, 2)]
            stability_rows.append(dict(dataset=ds, scope=scope, pairs=len(cor), median_spearman=float(np.median(cor))))
        print(f"{ds}: checked 20 saved NN ensembles; all 64 ranks and gradients reproduced.", flush=True)

    combined = pd.concat(all_rows, ignore_index=True)
    csv("all_rankings_long.csv", combined)
    csv("per_split_importance.csv", pd.DataFrame(split_rows))
    csv("per_lineage_importance.csv", pd.concat(lineage_rows, ignore_index=True))
    wide = None
    for ds in ["CTRP", "PRISM"]:
        d = combined.query("dataset == @ds").set_index("gene")
        keep = ["predictive_rank", "relative_mse_increase_pct", "relative_ci_low_pct", "relative_ci_high_pct",
                "gradient_rank", "mean_standardized_slope", "rms_standardized_slope", "higher_expression_predicts",
                "positive_splits", "positive_top10_splits", "positive_large_lineages", "eligible_large_lineages"]
        part = d[keep].add_prefix(ds+"__")
        wide = part if wide is None else wide.join(part)
    wide = wide.sort_values("CTRP__predictive_rank")
    csv("CTRP_PRISM_64_gene_comparison.csv", wide.reset_index())
    rho = float(spearmanr(wide.CTRP__predictive_rank, wide.PRISM__predictive_rank).statistic)
    summary = dict(genes=64, models=40, no_refitting=True, cross_assay_predictive_rank_spearman=rho,
        top10={ds:combined.query("dataset == @ds").head(10).gene.tolist() for ds in ["CTRP", "PRISM"]},
        positive_pointwise_relative_intervals={ds:combined.query("dataset == @ds and relative_ci_low_pct > 0").gene.tolist() for ds in ["CTRP", "PRISM"]},
        stability=stability_rows)
    dump("summary.json", summary)
    dump("environment.json", dict(python=sys.version, packages={name:importlib.metadata.version(name)
        for name in ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "matplotlib"]}))
    figures(combined)
    report(combined, summary)
    record(Path(__file__))
    dump("source_manifest.json", dict(files=list(records.values())))
    validation = dict(passed=True, registry_gene_order_exact=True, retained_models=40, ensemble_members=120,
        source_and_model_files_verified=len(records), predictive_ranks_exact=True, gradient_ranks_exact=True,
        positive_split_and_top10_counts_exact=True, relative_bootstrap_intervals_verified=True,
        no_refitting=True, no_panel_selection=True, train_test_identity_and_lineage_checks=True,
        max_absolute_errors=maxima, model_checks=checks,
        source_importance_validation=previous_validation)
    dump("validation.json", validation)
    dump("artifact_manifest.json", dict(files=[dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"]))
    print(json.dumps(dict(passed=True, max_absolute_errors=maxima, summary=summary), indent=2), flush=True)


def figures(frame):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.titleweight":"bold", "svg.fonttype":"none"})
    qa = []
    for count, height, filename in [(15, 8, "top15_predictive_importance"), (64, 21, "all64_predictive_importance")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, height))
        for ax, ds, color in zip(axes, ["CTRP", "PRISM"], ["#27678c", "#9b4e24"]):
            d = frame.query("dataset == @ds").head(count)
            yy = np.arange(len(d))
            ax.hlines(yy, d.relative_ci_low_pct, d.relative_ci_high_pct, color=color, linewidth=1.5, alpha=.6)
            ax.scatter(d.relative_mse_increase_pct, yy, c=color, s=28, zorder=3)
            ax.axvline(0, color="#555555", linewidth=.8, linestyle="--")
            ax.set_yticks(yy, [f"{r.predictive_rank:2d}  {r.gene}" for r in d.itertuples()])
            ax.set_ylim(len(d)-.4, -.6)
            ax.set_title(ds+"  |  retained 64-gene NN", loc="left", pad=13)
            ax.set_xlabel("Held-out MSE increase after replacement (%)")
            ax.grid(axis="x", color="#dddddd", linewidth=.5)
            ax.set_axisbelow(True)
        title = fig.suptitle("Gene importance for erastin sensitivity prediction", fontsize=16, weight="bold", y=.985)
        fig.text(.5, .017, "20 saved holdouts · exact same-lineage training-donor replacement\n"
                 "Bars: conditional pointwise 95% lineage-bootstrap intervals; no refitting", ha="center", fontsize=9)
        fig.subplots_adjust(top=.88 if count==15 else .945, bottom=.115 if count==15 else .06, left=.11, right=.98, wspace=.35)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            assert not title.get_window_extent(renderer).overlaps(ax._left_title.get_window_extent(renderer))
        qa.append(dict(figure=filename, genes_each=count, panel_and_main_titles_do_not_overlap=True))
        fig.savefig(OUT/(filename+".png"), dpi=160, facecolor="white")
        fig.savefig(OUT/(filename+".svg"), facecolor="white")
        plt.close(fig)
    dump("visual_qa.json", dict(title_overlap_checks_passed=True, figures=qa))


def report(frame, summary):
    lines = ["# All 64 genes in the retained neural network models", "",
        "The retained panel contains **64 genes**. "
        "These rankings use the exact retained, unweighted, expression-only CTRP and PRISM neural networks, separately. "
        "No new model was trained, no genes were selected or removed, and the retained-model registry was not changed.", "",
        "## How to read the rankings", "",
        "**Primary rank: held-out predictive importance.** For each held-out cell, replace one gene's expression with "
        "every outer-training donor value from the same lineage, keeping the other 63 genes fixed. Average squared-error "
        "changes over donors and receiver cells, then over the 20 saved lineage-stratified holdouts. Train/test identity "
        "groups are disjoint. All evaluated lineages occur in both train and test; this is a within-lineage generalization benchmark.", "",
        "**Importance (%) = 100 × mean increase in held-out MSE / mean baseline held-out MSE.** Larger positive values "
        "indicate greater useful reliance by this fitted model. Values do not sum to 100%, are not explained variance, "
        "and cannot be added to predict the benefit of a smaller panel. Negative values mean replacement improved "
        "held-out error under this procedure; they do not imply that a gene lacks ferroptosis relevance.", "",
        "**Secondary rank: effective input sensitivity.** A neural network has many hidden weights for each input; "
        "there is no single coefficient per gene. The gradient ranking uses the root mean square of the derivative "
        "of the three-member ensemble prediction with respect to standardized expression, evaluated on held-out cells "
        "across all 20 fits. Member gradients are averaged before taking magnitudes. Slopes are in original negative-AUC "
        "units per training-set expression SD. This incorporates all layers and activation functions. It measures "
        "how strongly expression changes the prediction, rather than whether that change is useful.", "",
        "The target is **negative erastin AUC** in both cohorts. Thus a positive signed slope means higher expression "
        "predicts greater erastin sensitivity; a negative slope means lower predicted sensitivity. Directions are "
        "local model associations with all other inputs held fixed, not experimental gene perturbations.", "",
        "## Main findings", "",
        "CTRP's top five genes are SLC7A11, ACSL4, GPX4, AIFM2 and ABCC1. PRISM's top five are AIFM2, SLC3A2, "
        "LRP8, SLC7A11 and SLC40A1. SLC7A11, AIFM2 and ACSL4 have positive pointwise lower uncertainty bounds "
        "in both cohorts. LRP8 is also top ten in both, but its PRISM interval includes zero.", "",
        f"Cross-assay predictive-rank Spearman correlation is **{summary['cross_assay_predictive_rank_spearman']:.3f}**. "
        "GPX4 ranks 3rd in CTRP versus 48th in PRISM; ABCC1 ranks 5th versus 27th. Their CTRP prominence is "
        "not reproduced by the PRISM model. Cohorts overlap in cell lines and differ in assay, response processing, "
        "sample composition and fitted models; rank differences cannot be assigned solely to assay biology.", ""]
    for ds in ["CTRP", "PRISM"]:
        stable = next(v for v in summary["stability"] if v["dataset"]==ds and v["scope"]=="between_large_lineages")
        lines += [f"{ds}: genes with a positive pointwise lower 95% bound are **" + ", ".join(summary["positive_pointwise_relative_intervals"][ds]) + "**. "
                  f"Median pairwise predictive-rank correlation across lineages with at least 10 cohort cells is {stable['median_spearman']:.3f}. "
                  "Importance is therefore not a universal lineage ranking.", ""]
    lines += ["![Top 15 predictive genes](top15_predictive_importance.png)", "",
        "## Ranking by effective NN input sensitivity", "",
        "RMS standardized gradient is the weights-like measure described above, in negative-AUC units per expression SD. "
        "It has no sign; the predictive tables below and CSVs supply signed directions. The two importance definitions "
        "can disagree: HDDC3 ranks 7th by PRISM gradient magnitude but 54th by predictive importance. A strong effect "
        "on predictions can be unhelpful for predicting the measured response.", "",
        "| Gradient rank | CTRP gene | CTRP RMS gradient | PRISM gene | PRISM RMS gradient |",
        "|---:|---|---:|---|---:|"]
    c = frame.query("dataset == 'CTRP'").sort_values("gradient_rank").head(10)
    p = frame.query("dataset == 'PRISM'").sort_values("gradient_rank").head(10)
    for cr, pr in zip(c.itertuples(), p.itertuples()):
        lines.append(f"| {cr.gradient_rank} | {cr.gene} | {cr.rms_standardized_slope:.6f} | {pr.gene} | {pr.rms_standardized_slope:.6f} |")
    lines += ["",
        "## Complete predictive ranking", "",
        "Interval = conditional pointwise 95% whole-lineage bootstrap interval for relative MSE increase. "
        "Positive / top-10 counts refer to the 20 overlapping saved holdouts; top-10 counts also require positive importance. "
        "Lineage counts are positive importance among lineages with at least 10 cohort cells (19 CTRP; 13 PRISM). "
        "Higher / lower refers to predicted sensitivity when expression rises. The complete CSVs additionally contain "
        "signed and RMS slopes, direction agreement, rank ranges, and common-lineage summaries.", ""]
    for ds in ["CTRP", "PRISM"]:
        lines += [f"### {ds}", "", "| Rank | Gene | MSE increase % [95% interval] | Higher expression predicts | Positive /20 | Top 10 /20 | Positive lineages | Gradient rank |", "|---:|---|---:|---|---:|---:|---:|---:|"]
        for r in frame.query("dataset == @ds").itertuples():
            direction = "higher sensitivity" if r.mean_standardized_slope>0 else "lower sensitivity"
            lines.append(f"| {r.predictive_rank} | {r.gene} | {r.relative_mse_increase_pct:.3f} [{r.relative_ci_low_pct:.3f}, {r.relative_ci_high_pct:.3f}] | {direction} | {r.positive_splits} | {r.positive_top10_splits} | {r.positive_large_lineages}/{r.eligible_large_lineages} | {r.gradient_rank} |")
        lines += ["", f"[Complete {ds} predictive ranking]({ds.lower()}_64_gene_ranking.csv) · "
                  f"[{ds} gradient-sorted ranking]({ds.lower()}_64_gene_gradient_ranking.csv)", ""]
    lines += ["## Interpretation limits and checks", "",
        "Intervals use the existing 5,000 whole-lineage bootstrap draws with models, donors and panel fixed. "
        "They are not corrected for multiple genes or previous model/panel comparisons and do not include retraining uncertainty. "
        "Twenty overlapping holdouts are not independent biological replications. Treat these as exploratory explanations of the "
        "retained models, rather than a fresh gene-discovery test.", "",
        "Replacement is marginal within lineage and can disrupt correlations between a gene and the other inputs. Correlated "
        "genes can share or mask importance. Gene importance describes a particular fitted model rather than an intrinsic "
        "gene property; see the [official scikit-learn explanation](https://scikit-learn.org/1.5/modules/permutation_importance.html). "
        "Our donor replacement is exhaustive training-donor replacement, not a call to sklearn's random permutation routine.", "",
        "Validation checks exact registry and source hashes; all 40 saved ensembles and their train/test IDs; all 64 gene orders; "
        "all held-out predictions and analytic ensemble gradients; finite-difference derivatives and direct exhaustive donor "
        "replacement for all 64 genes in one receiver from each cohort's first and last holdout; aggregate predictive ranks; "
        "gradient ranks; split stability counts; and the saved relative bootstrap intervals. The original importance analysis "
        "also validated donor identity exclusions and its complete replacement calculation. See [validation.json](validation.json).", "",
        "Files: [both cohorts side by side](CTRP_PRISM_64_gene_comparison.csv), "
        "[per-lineage effects](per_lineage_importance.csv), [per-split effects](per_split_importance.csv), "
        "[all-64 figure](all64_predictive_importance.png), [source hashes](source_manifest.json).", "",
        "Reproduce from the project root with the original pinned Python environment:", "", "```bash",
        'PYTHONPATH="$PWD/work/bulkformer/packages" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache" PYTHONDONTWRITEBYTECODE=1 \\',
        '/home/sji/miniforge3/envs/deeplearning/bin/python outputs/retained_nn_gene_ranking/rank_genes.py',
        "```", "", "Generated UTC: " + datetime.now(timezone.utc).isoformat(timespec="seconds"), ""]
    (OUT / "REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
