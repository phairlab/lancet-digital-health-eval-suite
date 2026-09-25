# Confidence Intervals

[← back to the README](../README.md)

Two mechanisms are available, and they answer **different questions**. Reporting the right one — or both — matters more than the choice of software.

| | Bootstrap CI | Nadeau-Bengio CI |
|---|---|---|
| Flag | `--bootstrap N` | `--bengio-correction` |
| Applies to | One model's performance | The *difference* between two models |
| Source of uncertainty | Which subjects you happened to sample | Which training split you happened to get |
| Unit of analysis | Subject (or cluster) | Fold |
| Where it lands | `bootstrap_ci.json`, `combined_metrics*` | `bengio_correction.csv` |
| Typical paper use | Per-model results table | Model-comparison claims |

A bootstrap interval will usually be **narrow** on a large dataset, because it reflects patient-sampling noise only and says nothing about how much the model would change if retrained on a different split. The Nadeau-Bengio interval captures the latter but is based on only `k` fold scores, so it has few degrees of freedom. They are complements, not competitors. If they disagree wildly, something is wrong — that is a useful cross-check.

## Bootstrap CIs on model performance

```bash
python ldh_eval.py --input_dir "experiment_results/" --bootstrap 2000
```

This resamples the pooled out-of-fold predictions with replacement `N` times, recomputing every rate-valued metric on each resample, and reports the percentile interval. Metrics covered: `prevalence_pct`, `auroc`, `calibration_slope`, `brier_score`, and — when `--threshold` is given — `alert_rate`, `sensitivity`, `specificity`, `ppv`, `npv`.

Raw counts (`n`, `tp`, `tn`, `fp`, `fn`) are deliberately **excluded**: a count computed on a resample is not comparable to the count on the original data.

Note that no models are retrained. The bootstrap operates entirely on the probabilities you already saved, which is why it costs seconds rather than days. (The `.632+` and Harrell optimism-correction bootstraps *do* refit per resample — that is a different procedure, and not what this implements.)

## Clustered data (IMPORTANT)

If **one unit can contribute more than one row** to your dataset, rows are not independent and the default row-level bootstrap will produce intervals that are **too narrow**. This is extremely common in medical data:

* a patient with several hospital encounters
* a subject imaged at several timepoints
* an eye, joint, or lesion measured repeatedly
* a recruiting site contributing many cases

The fix is the **cluster bootstrap**: resample *units* with replacement, taking all of a unit's rows together. Supply a per-row cluster label and this happens automatically.

**How much does it matter?** Variance inflation tracks the *effective* cluster size, `Σm²/Σm`, not the mean — a long tail of large clusters matters far more than the average. On a real 121k-row cohort with 65k units, mean 1.86 rows per unit but a heavy tail, the naive interval was too narrow by **1.43× for AUROC and 1.80× for PPV**. Rank-based measures like AUROC are somewhat buffered; metrics that are row means (PPV, alert rate, prevalence) are hit hardest.

The tool reports the diagnostics so you can see this for yourself:

```
4,000 clusters | mean size 2.23 | effective size 2.75 | max 8
```

There are two ways to supply the labels.

**Route 1 — the label is already in the prediction JSON.** Preferred. Auto-detected if the key is named `subject_ids`, `patient_ids`, `cluster_ids`, or `group_ids`:

```bash
python ldh_eval.py --input_dir "experiment_results/" --bootstrap 2000
```

Name it explicitly if you use some other key:

```bash
python ldh_eval.py --input_dir "experiment_results/" --bootstrap 2000 --cluster-key eye_id
```

**Route 2 — join the label from a CSV.** Use this when your saved predictions carry a *record*-level id (an encounter, admission, or image id) but the cluster is a level above it, and you don't want to re-run inference just to add a column. Supply a two-column lookup:

```csv
record_id,subject_id
10001234,7001
10001235,7001
10002000,7002
```

```bash
python ldh_eval.py --input_dir "experiment_results/" --bootstrap 2000 \
    --id-key record_ids \
    --cluster-map lookup.csv \
    --cluster-map-cols record_id,subject_id
```

`--cluster-map-cols` defaults to the first two columns of the file in `(record, cluster)` order, so name them explicitly if your CSV happens to list them the other way round. Numeric ids are normalised, so `12345`, `"12345"`, and `12345.0` all match.

Every row must resolve to a cluster. If any record id is missing from the map the run **stops with an error** rather than silently mixing clustered and unclustered rows.

**If no cluster label is found**, the tool falls back to row-level resampling and prints a warning. That fallback is correct only when each unit contributes exactly one row.

## Nadeau-Bengio CIs on model differences

Included automatically with `--bengio-correction`. The interval is built from the *same* corrected variance estimate as the p-value, so it excludes zero exactly when `p < 1 - ci_level`. Console output looks like:

```
Pairwise AUROC comparisons (Bengio-Nadeau corrected, two-tailed, 95% CI):
  * A0 vs A1: Δ=+0.0932 (+0.0463 to +0.1401), p=0.00528
```

Reportable as: *"AUROC difference +0.093 (95% CI 0.046 to 0.140)"*. For every metric, `bengio_correction.csv` carries `{metric}_ci_lo`, `{metric}_ci_hi`, and a ready-to-paste `{metric}_formatted` column.

Clustering does **not** affect this interval, provided each cluster stays within a single fold (as it must, to avoid leakage) — the fold is the unit of analysis, not the row.

## A note on the Bengio-Nadeau correction

A standard paired t-test over fold-level differences underestimates variance because any two folds share the majority of their training data, making their scores positively correlated. Nadeau & Bengio (1999) derived a corrected variance estimate:

```
corrected_var = (1/k + n_test/n_train) × var(differences)
```

where `k` is the number of folds, `n_test` is the average test set size per fold, and `n_train` is the average training set size per fold. The t-statistic is then `mean(differences) / sqrt(corrected_var)`, evaluated against a t-distribution with `k - 1` degrees of freedom. The correction is conservative relative to the naive test, and is stronger when `n_test/n_train` is large — that is, when you use few folds.

`n_train` is read from your `train_*_predictions.json` files when they are present, and the `n_train_source` column in `bengio_correction.csv` records `measured` when this happened. If they are absent it falls back to the standard k-fold identity `n_train ≈ (k-1) × n_test` and reports `assumed`. Note that under that fallback the correction factor reduces to `1/k + 1/(k-1)`, which depends only on the fold count — dataset size cancels out entirely. Saving the training predictions is therefore worthwhile even if you never use `--recalibrate`.

## Two caveats worth stating in a methods section

1. **No multiple-comparison correction is applied.** Comparing 7 experiments means 21 pairwise tests per metric. If you report `sig_p05` across many pairs, apply Holm or Benjamini-Hochberg yourself, or say plainly that the p-values are uncorrected.
2. **The bootstrap CI does not include model-training variability**, and the Nadeau-Bengio CI does not include patient-sampling variability. Neither is a complete uncertainty estimate on its own.

## Runtime

Roughly 25 ms per resample at 120k rows for all metrics combined, single-threaded:

| Rows | `--bootstrap 2000`, per experiment |
|---|---|
| 10k | ~5 s |
| 120k | ~50 s |
| 120k × 7 experiments | ~6 min |

The calibration slope is the most expensive component (it fits a logistic regression per resample). Halve `N` to halve the time; `N=1000` is usually enough for a stable 95% interval, while `N=2000+` steadies the tails.


## What the `±` means, and what it does not

`_std` and the `(±…)` strings are the **sample standard deviation across folds** (`ddof=1`). This is a description of *spread*: how much the metric moved from fold to fold. It is deliberately **not** a confidence interval, and the two are easy to confuse in a results table.

For 10 folds of a metric with a fold SD of 0.021:

| Quantity | Value | What it answers |
|---|---|---|
| SD across folds | 0.021 | How much did performance vary between folds? Does **not** shrink as you add folds. |
| SE of the mean (`SD/√k`) | 0.007 | How precisely is the average pinned down? Shrinks with more folds — **but is anti-conservative for CV**, because folds share training data and their scores are correlated. |
| Bootstrap 95% CI | ±0.017 | What range plausibly contains the true value, given patient sampling? |

Three consequences worth knowing:

* **Never report `SD/√k` as a confidence interval for cross-validated performance.** It is roughly 1.45× too narrow at k=10. Use the bootstrap CI (`--bootstrap`) for single-model performance, or the Nadeau-Bengio interval for differences between models — the latter applies exactly this correction.
* **If you report the `±` columns, label them "SD across folds"**, not "±95% CI" and not "±SE". A reader who assumes a CI will draw the wrong conclusion about precision.
* **At k=10 the two happen to look similar** — the Nadeau-Bengio 95% half-width works out to about 1.1 × the fold SD, so `mean ± 1 SD` coincidentally approximates a corrected 95% interval. This is an artefact of k=10, not a rule; do not rely on it.

With a single fold there is no sample to take an SD from, so `_std` is `nan` rather than `0`.

## CIs on the plots

When `--bootstrap N` is set, the plots that report numbers carry the intervals too:

| Plot | Label |
|---|---|
| `pooled_auroc.png` | `AUROC = 0.887 (95% CI 0.866 to 0.906)` in the legend |
| `pooled_calibration.png` | Slope and Brier CIs in the title (two lines) |
| `overlay_auroc.png` | `name (AUC=0.887, 95% CI 0.866-0.906)` per experiment |
| `overlay_calibration.png` | `name (slope=1.000, 95% CI 0.911-1.094)` per experiment |

Two deliberate choices here:

* **The overlay ROC legend reports the pooled AUROC, not the fold mean.** The curve on those axes *is* the pooled curve, so labelling it with the average of the per-fold AUROCs would put a different estimand in the legend from the line being drawn. The two differ because AUROC is a rank statistic: the fold mean only counts (positive, negative) pairs within the same fold, while the pooled figure counts every pair — typically around 90% of which span different folds. Without `--bootstrap`, the legend falls back to the fold mean and says so explicitly (`fold mean AUC=0.891±0.021 SD`).
* **Per-fold plots never carry CIs**, and **decision curves never carry CIs**. Per-fold plots have no bootstrap of their own; decision curves are omitted because Van Calster et al. advise against attaching confidence intervals to clinical utility measures, where quantifying uncertainty remains contested.
