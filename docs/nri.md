# Net Reclassification Improvement (NRI)

[← back to the README](../README.md)

`nri.py` computes two-category NRI at a fixed decision threshold, comparing one or more models against a common baseline.

**Van Calster et al. do not recommend it, and neither do we.** NRI is an improper scoring rule: a miscalibrated model can score better than a well-calibrated one, and the statistic is not centred at zero under the null when either model is misspecified, so an NRI above zero is not evidence that the new model adds information. Use the decision curve for clinical utility and a likelihood-ratio test for whether a predictor earns its place. NRI is here because clinical collaborators and reviewers ask for it routinely — and if it is going to be reported, it should at least be computed on the same probability scale, the same folds, and the same resampling unit as everything else in the table. Report it *beside* the decision curve, never instead of it.

At threshold *t*, positive values favouring the new model:

```
event component     = (events up − events down) / n_events           = Δ sensitivity
non-event component = (non-events down − non-events up) / n_non_events = Δ specificity
NRI                 = event component + non-event component
```

Both identities are asserted at runtime, so a counting error fails loudly rather than silently.

## Usage

```bash
python nri.py --baseline_dir results/baseline --comparison_dirs results/model_a results/model_b \
              --threshold 0.20 --recalibrate --bootstrap 2000 \
              --cluster-key patient_id --output_csv results/nri.csv
```

`--ordering` accepts the same JSON file as `ldh_eval.py` and supplies the comparison directories, their labels, and their order; the baseline is excluded automatically. It is mutually exclusive with `--comparison_dirs`/`--labels`.

Or from Python:

```python
from nri import compare_experiments, format_nri_table

df = compare_experiments('results/baseline', ['results/model_a'], threshold=0.20,
                         recalibrate=True, n_boot=2000)
print(format_nri_table(df).to_string())
```

## What it shares with `ldh_eval.py`

* **The same prediction files.** No new format: it reads the same `fold_*_predictions.json`, and `train_*_predictions.json` when `--recalibrate` is set.
* **The same recalibration map**, via `fit_apply_recalibration()` in `core_eval_functions.py`, fitted per fold on that fold's training predictions.
* **The same cluster bootstrap.** `--cluster-key`, `--id-key`, `--cluster-map`, and `--cluster-map-cols` behave exactly as documented under [Clustered data](confidence-intervals.md#clustered-data-important), and the same warning applies: without cluster labels the interval assumes one row per subject. The bootstrap is *paired* — one set of resampled indices is applied to both models, preserving the within-patient correlation.

## Three things to get right

* **`--recalibrate` must match whatever you used for the threshold metrics.** Recalibration is monotone within a fold, so it leaves that fold's AUROC untouched but moves patients across a fixed absolute-risk threshold. Computing NRI on raw probabilities while reporting sensitivity and specificity on recalibrated ones produces two tables that appear to contradict each other — a model can beat the baseline on both sensitivity and specificity in one table while showing a negative NRI in the other.
* **NRI is paired, so both experiments must have scored the same patients in the same order.** This is checked on per-row record ids when the prediction files carry any recognised id key, and falls back to comparing outcome vectors otherwise — weaker, because it cannot detect a permutation that preserves the outcome sequence, so it warns. A mismatch raises rather than returning a meaningless number.
* **Report the event and non-event components separately**, which `format_nri_table()` does by default. A positive total built from a large gain in events and a loss in non-events means something quite different from a small gain in both, and the total alone hides it.

The returned DataFrame carries the pooled NRI with its bootstrap CI, both components, the raw reclassification counts, and the fold mean and SD (`ddof=1`, same convention as everywhere else — see [What the `±` means](confidence-intervals.md#what-the--means-and-what-it-does-not)).
