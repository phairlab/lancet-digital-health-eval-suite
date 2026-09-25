# PHAIR Model Evaluation Suite: Medical Binary Classification

Quick, rigorous evaluation of clinical prediction models, following the core recommendations from Van Calster et al.'s 2025 *Lancet Digital Health* Viewpoint. Given predicted probabilities and true outcomes, it produces the four essential plots and numbers:

1. **AUROC** — discrimination performance
2. **Calibration plot** — agreement between predictions and observations, with loess smoothing
3. **Decision curve** — net benefit across decision thresholds
4. **Risk distribution** — probability distributions by outcome, as violin plots

Performance metric selection for clinical ML is contentious, and researchers disagree on which measures are appropriate. This repo implements the recommendations of a comprehensive expert consensus paper, giving you a defensible, citable rationale for your evaluation approach.

## Contents

* [Installation](#installation)
* [Getting started](#getting-started) — save your predictions, then run the evaluation
* [Command-line arguments](#command-line-arguments)
* [Output files](#output-files)
* [Using it as a library](#using-it-as-a-library)
* [Interpreting the plots](#interpreting-the-plots)

Deeper reference material lives in `docs/`:

| Document | Read it when |
|---|---|
| [Confidence intervals](docs/confidence-intervals.md) | You are reporting uncertainty — which interval answers which question, why clustered data needs special handling, and what the `±` columns do and do not mean |
| [Net Reclassification Improvement](docs/nri.md) | A collaborator or reviewer has asked for NRI |
| [Troubleshooting](docs/troubleshooting.md) | An experiment failed to evaluate, or you are wondering why the run stopped |

## Installation

```bash
python3 -m venv ldh_eval
source ldh_eval/bin/activate
pip install -r requirements.txt
```

or

```bash
conda create -n ldh_eval
conda activate ldh_eval
conda install --yes --file requirements.txt
```

I recommend having a single environment for this evaluation that you can activate across multiple projects when it's time for reporting.

## Getting started

The recommended workflow is the command line over cross-validation folds. It gives you per-fold results, pooled results, and — with `--recurse` — overlay plots comparing several experiments.

### 1. Save your predictions

In the code that trains or runs inference on your model, save the true classes and predicted probabilities as JSON:

```python
import os
import json

# TRAINING LOOP
for fold_num in range(n_folds):

    ##### ...
    ##### Training logic 
    ##### ...

    # EXAMPLE: FETCH MODEL PROBABILITIES
    test_probas_ = classifier.predict_proba(X_test)
    train_probas_ = classifier.predict_proba(X_train)

    # VERIFY OUTPUT DIRECTORY
    results_folder_path = "experiment_results"
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # SAVE TEST CLASSES AND PROBABILISTIC PREDICTIONS FROM MODEL
    test_predictions = {
        'y_true': y_test.tolist(),
        'y_proba': test_probas_[:, 1].tolist(),

        # OPTIONAL, but required for cluster-bootstrap confidence intervals.
        # Include this whenever one unit (patient, subject, eye, site) can
        # contribute more than one row.
        'subject_ids': subject_ids_test.tolist(),
    }
    with open(f"{results_folder_path}/fold_{fold_num}_predictions.json", 'w') as f:
        json.dump(test_predictions, f, indent=4)

    # SAVE TRAIN CLASSES AND PROBABILISTIC PREDICTIONS FROM MODEL
    train_predictions = {
        'y_true': y_train.tolist(),
        'y_proba': train_probas_[:, 1].tolist()
    }
    with open(f"{results_folder_path}/train_{fold_num}_predictions.json", 'w') as f:
        json.dump(train_predictions, f, indent=4)

```

Saving the training predictions is worth doing even if you never use `--recalibrate`: they let the Nadeau-Bengio correction use your measured training-set size instead of assuming it.

**Recognised keys:**

| Key | Required | Purpose |
|---|---|---|
| `y_true` | yes | Binary outcomes |
| `y_proba` | yes | Predicted probabilities for the positive class |
| `subject_ids` / `patient_ids` / `cluster_ids` / `group_ids` | no | Per-row cluster label, auto-detected for the cluster bootstrap |
| `record_ids` / `encounter_ids` / `admission_ids` / `ids` | no | Per-row record id, auto-detected when joining cluster labels from a CSV via `--cluster-map` |

Any other keys (`y_pred`, `test_indices`, and so on) are ignored, so it is safe to save extra fields. Names are only auto-detected — you can use any key name you like and point at it explicitly with `--cluster-key` or `--id-key`.

### 2. Run the evaluation

```bash
python ldh_eval.py --input_dir "experiment_results/"
```

This saves the in-fold plots and numbers in each fold's sub-folder, and the meta-analysis of all folds into `--input_dir` itself.

The full analysis, with cluster-aware confidence intervals:

```bash
python ldh_eval.py --input_dir "experiment_results/" --recurse --recalibrate \
    --threshold "0.2" --bengio-correction --ordering example_ordering.json --bootstrap 2000
```

**If one unit can contribute more than one row to your data** — a patient with several encounters, an eye imaged repeatedly, a recruiting site — read [Clustered data](docs/confidence-intervals.md#clustered-data-important) before reporting any interval. Without a cluster label the intervals will be too narrow, on real cohorts by as much as 1.8×.

## Command-line arguments

| Argument | Requires | Effect |
|---|---|---|
| `--input_dir DIR` | — | Directory containing the prediction JSONs. The only required argument. |
| `--recurse` | — | Treat `DIR` as containing several experiments, each with their own folds. Evaluates each, then draws overlay ROC, calibration, and decision curves across all of them. |
| `--recalibrate` | `train_*` files | **Recommended.** Logistic recalibration to straighten the calibration curve. Monotone within a fold, so discrimination is unaffected, but calibration metrics improve with few drawbacks. The only option that reads the `train_*_predictions.json` files. |
| `--threshold T` | — | Adds threshold-dependent metrics at `T` (between 0 and 1): `sensitivity`, `specificity`, `ppv`, `npv`, `tp`, `tn`, `fp`, `fn`, and `alert_rate` — the percentage of predictions that are positive at that threshold. |
| `--bootstrap N` | — | Bootstrap confidence intervals from `N` resamples of the pooled out-of-fold predictions. Default `0` (off); `2000` is reasonable for reporting. See [Confidence intervals](docs/confidence-intervals.md). |
| `--bengio-correction` | `--recurse` | Pairwise Nadeau-Bengio corrected t-tests between all experiments, with corrected CIs on each difference. |
| `--ordering FILE` | `--recurse` | JSON file pinning which experiments appear in overlays, in what order, under what label. See below. |
| `--cluster-key KEY` | — | Names the per-row cluster label in your JSONs. Only needed when auto-detection misses it. |
| `--cluster-map FILE.csv` | — | Supplies cluster labels from an external CSV, joined on a record id. Pair with `--id-key KEY` and optionally `--cluster-map-cols ID,CLUSTER`. |
| `--ci-level L` | — | Confidence level for both interval types (default `0.95`). |
| `--seed S` | — | Seeds bootstrap resampling for reproducible intervals (default `0`). |
| `--skip-failed` | `--recurse` | Downgrades a failed experiment from an error to a warning. See [Troubleshooting](docs/troubleshooting.md) — you almost certainly do not want this for figures you intend to publish. |

**`--ordering`** points at a JSON file mapping experiment directory names to plot labels. This pins the order experiments appear in, gives them short legend labels, and restricts the overlays to just the listed experiments. See `example_ordering.json`:

```json
{
    "D1_removedtop0percent": "D1: Remove Top 0%",
    "D1_removedtop5percent": "D1: Remove Top 5%"
}
```

A list of `[directory_name, plot_label]` pairs is also accepted. Directories named in the file but absent from disk are skipped with a note; if none of them match, the script stops and lists what it expected against what it found. Without `--ordering`, every subdirectory is included, ordered alphabetically and labelled by directory name.

## Output files

| File | Location | Contents |
|---|---|---|
| `metrics.json` | each fold dir | Per-fold metrics, including `n_train` when available |
| `aggregate_metrics.json` | experiment dir | Fold mean ± sample SD per metric |
| `bootstrap_ci.json` | experiment dir | Point estimate, CI bounds, bootstrap SD, and cluster diagnostics |
| `combined_metrics.csv` | `overlay_results/` | Numeric table: `_mean`, `_std`, plus `_pooled`, `_ci_lo`, `_ci_hi` |
| `combined_metrics_formatted.tsv` | `overlay_results/` | Paste-ready: `mean (±SD)` columns plus `{metric}_ci` columns |
| `bengio_correction.csv` | `overlay_results/` | Pairwise differences, CIs, t-statistics, p-values |
| `bengio_correction_auroc_pvals.csv` | `overlay_results/` | Square AUROC p-value matrix |

The `_mean`/`_std` columns are never removed or overwritten, so existing tables built on them keep working. Bear in mind that `{metric}_mean` (average of the per-fold values) and `{metric}_pooled` (computed once on all out-of-fold predictions) are different estimands and will differ slightly; the CI belongs to the pooled figure.

p-values are written at full float precision rather than rounded, so genuinely tiny values appear as e.g. `4.99e-06` instead of collapsing to `0.0`.

**The `±` columns are the SD across folds, not a confidence interval.** If you report them, label them that way — see [What the `±` means](docs/confidence-intervals.md#what-the--means-and-what-it-does-not).

## Using it as a library

For a single set of predictions with no cross-validation:

```python
from core_eval_functions import *
from ldh_eval import evaluate_model

# Individual plots
auroc(y_true, y_prob)
calibration(y_true, y_prob)
decision_curve(y_true, y_prob, threshold_range=(0.0, 0.5))
risk_distribution(y_true, y_prob)

# All at once
evaluate_model(y_true, y_prob, output_dir="results/")
```

## Interpreting the plots

### AUROC (ROC curve)
The AUROC quantifies discrimination—the model's ability to rank patients correctly. Values closer to 1.0 indicate better discrimination; 0.5 is random guessing.

<img src="examples/overlay_auroc.png" alt="AUROC Plot" width="500">

### Calibration plot
The calibration plot shows whether predicted probabilities match observed outcomes. The loess curve should hug the diagonal; deviations indicate the model systematically over- or under-predicts risk. A calibration slope near 1.0 is ideal (slope < 1 suggests overfitting, slope > 1 suggests underfitting). Poor calibration can often be solved by using the `--recalibrate` argument.

<img src="examples/overlay_calibration.png" alt="Calibration Plot" width="500">

### Decision curve analysis
Decision curves show net benefit—whether using the model improves decisions compared to "treat all" or "treat none" strategies. The model is clinically useful only where its curve (blue) is above both reference lines; higher net benefit is better.

<img src="examples/overlay_decision_curve.png" alt="Decision Curve Plot" width="500">

### Risk distribution by outcome
Violin plots show how predicted probabilities are distributed for patients who did vs. didn't experience the outcome. Good discrimination means clear separation: negatives clustered at low probabilities, positives clustered at high probabilities, with minimal overlap.

<img src="examples/overlay_risk_distribution.png" alt="Risk Distribution Plot" width="500">

When `--bootstrap` is set, the plots that report numbers carry their intervals too — see [CIs on the plots](docs/confidence-intervals.md#cis-on-the-plots).

## Citation for Original Paper

Van Calster B, Collins GS, Vickers AJ, Wynants L, Kerr KF, Barreñada L, Varoquaux G, Singh K, Moons KGM, Hernandez-Boussard T, Timmerman D, McLernon DJ, van Smeden M, Steyerberg EW, on behalf of Topic Group 6 of the STRATOS initiative. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *Lancet Digit Health* 2025. https://doi.org/10.1016/j.landig.2025.100916
