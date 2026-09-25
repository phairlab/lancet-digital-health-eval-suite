# Troubleshooting

[← back to the README](../README.md)

## When an experiment fails

With `--recurse`, if any experiment fails to evaluate the run **stops with an error** and no overlay plots or comparison tables are written. This is deliberate. The alternative — carrying on and emitting a figure that quietly omits a model — is the worst possible outcome, because the output looks complete and publishable.

The error names every failure with its cause, and lists what did succeed:

```
RuntimeError: 2 of 7 experiment(s) failed to evaluate:
  - LACE: KeyError: ".../LACE/1/fold_1_predictions.json: --id-key 'hadm_ids' not
    present. Available keys: ['test_indices', 'y_pred', 'y_proba', 'y_true']"
  - LACE-C: KeyError: ...

Succeeded: ['A0', 'A1', 'AH', 'D0', 'D1']

Stopping rather than writing overlay plots and comparison tables that silently omit
the failed experiment(s) ...
```

Common causes: a missing `train_*_predictions.json` under `--recalibrate`, a cluster key present in some experiments but not others, or a record id absent from `--cluster-map`.

Three cases are treated differently, on purpose:

| Situation | Behaviour |
|---|---|
| Experiment exists but fails to evaluate | **Error** — the run stops |
| Auto-discovered subdirectory containing no `fold_*_predictions.json` | Skipped with a note — it isn't an experiment (e.g. a `notes/` folder) |
| Directory named in `--ordering` that doesn't exist on disk | Skipped with a note — lets one ordering file be reused across subsets |

Note the last row: a *missing* directory is a note, whereas a directory that exists and *breaks* is an error. That way a shared ordering file can list every experiment you have ever run, while a real failure in a run you actually asked for still stops you.

`--skip-failed` reverses the first case, continuing with whatever succeeded and printing a prominent warning naming what was excluded. Reserve it for exploratory runs.
