#!/usr/bin/env python3
"""
Net Reclassification Improvement (NRI) at a fixed decision threshold.

Companion module to the PHAIR Model Evaluation Suite. Consumes the same
fold_*_predictions.json / train_*_predictions.json files as ldh_eval.py, applies
the identical recalibration map, and reuses the same cluster-bootstrap
machinery, so NRI lands on the same probability scale and the same resampling
unit as the threshold metrics, decision curves, and calibration slopes produced
by evaluate_cross_validation().

That alignment is the point. Recalibration is monotone within a fold, so it
leaves that fold's AUROC untouched, but it moves patients across a fixed
absolute-risk threshold. Computing NRI on raw probabilities while reporting
sensitivity and specificity on recalibrated ones produces two tables that
appear to contradict each other: a model can show higher sensitivity AND higher
specificity than the baseline in one table while showing a negative NRI in the
other.

(Pooled AUROC can still drift in the third decimal or beyond, because each fold
gets its own recalibration fit, so the pooled probability vector is piecewise
monotone rather than globally monotone. Threshold metrics move considerably
more.)

Two-category NRI at threshold t, comparing a new model against a baseline:

    event component     = (events moving up - events moving down) / n_events
                        = sensitivity_new - sensitivity_baseline
    non-event component = (non-events moving down - non-events moving up) / n_non_events
                        = specificity_new - specificity_baseline
    NRI                 = event component + non-event component

Both identities are asserted at runtime as a self-check.

Written for Sacha Davis (sdavis1@ualberta.ca).
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core_eval_functions import fit_apply_recalibration
from ldh_eval import (
    resolve_input_dir, extract_cluster_ids, load_cluster_map, load_ordering,
    _norm_id, DIRECT_CLUSTER_KEYS, RECORD_ID_KEYS,
)


# ============================================================================
# LOADING
# ============================================================================

def _fold_files(experiment_dir: Path) -> List[Path]:
    """Fold prediction files, sorted identically to evaluate_cross_validation."""
    files = sorted(experiment_dir.rglob('fold_*_predictions.json'))
    if not files:
        raise ValueError(f"No fold_*_predictions.json files found in {experiment_dir}")
    return files


def _row_ids(data: dict) -> Optional[np.ndarray]:
    """Per-row identifiers, if the prediction file carries any.

    Used to verify that two experiments scored the same patients in the same
    order. Searches the same keys ldh_eval uses for clustering, so no new file
    format is required.
    """
    key = next((k for k in RECORD_ID_KEYS + DIRECT_CLUSTER_KEYS if k in data), None)
    if key is None:
        return None
    return np.array([_norm_id(v) for v in data[key]], dtype=object)


def load_experiment(experiment_dir: str | Path,
                    recalibrate: bool = True,
                    cluster_key: Optional[str] = None,
                    id_key: Optional[str] = None,
                    cluster_map: Optional[Dict[str, str]] = None,
                    want_clusters: bool = False) -> dict:
    """Load per-fold out-of-fold predictions for one experiment.

    Mirrors evaluate_cross_validation(): with recalibrate=True the logistic
    recalibration is fitted on that fold's TRAINING predictions and applied to
    its held-out test predictions, so the returned probabilities are genuinely
    out-of-sample and carry no optimism.

    Returns a dict with 'y_true', 'y_prob' (lists of per-fold arrays),
    'row_ids' (list of per-fold arrays or None), and 'cluster_ids'
    (pooled array or None).
    """
    experiment_dir = resolve_input_dir(str(experiment_dir))
    y_true_folds, y_prob_folds, id_folds = [], [], []
    pooled_clusters, clusters_available = [], want_clusters

    for fold_idx, test_file in enumerate(_fold_files(experiment_dir)):
        with open(test_file) as f:
            test = json.load(f)
        y_true = np.asarray(test['y_true'])
        y_prob = np.asarray(test['y_proba'], dtype=float)

        if recalibrate:
            train_file = test_file.with_name(
                test_file.name.replace('fold_', 'train_', 1))
            if not train_file.exists():
                raise FileNotFoundError(
                    f"recalibrate=True needs training predictions, but "
                    f"'{train_file.name}' was not found in {train_file.parent}. "
                    f"Either save train_*_predictions.json alongside your "
                    f"fold_*_predictions.json (see README), or pass recalibrate=False."
                )
            with open(train_file) as f:
                train = json.load(f)
            y_prob = fit_apply_recalibration(
                np.asarray(train['y_true']),
                np.asarray(train['y_proba'], dtype=float),
                y_prob,
            )

        if clusters_available:
            ids = extract_cluster_ids(
                test, y_true.size, source=str(test_file),
                cluster_key=cluster_key, id_key=id_key, cluster_map=cluster_map,
                announce=(fold_idx == 0),
            )
            if ids is None:
                clusters_available = False
            else:
                pooled_clusters.extend(ids)

        y_true_folds.append(y_true)
        y_prob_folds.append(y_prob)
        id_folds.append(_row_ids(test))

    return {
        'y_true': y_true_folds,
        'y_prob': y_prob_folds,
        'row_ids': id_folds,
        'cluster_ids': (np.array(pooled_clusters, dtype=object)
                        if clusters_available and pooled_clusters else None),
        'name': experiment_dir.name,
    }


def _check_alignment(base: dict, comp: dict) -> str:
    """NRI is a paired comparison: both models must score the same patients, in
    the same order, fold by fold.

    Prefers per-row record ids when the prediction files carry them. Falls back
    to comparing outcome vectors, which is weaker (it cannot detect a permutation
    that happens to preserve the outcome sequence) and therefore warns.

    This is not cosmetic. If fold assignment or row order ever differs between
    two experiment directories, every number downstream is silently meaningless.
    """
    a_name, b_name = base['name'], comp['name']
    if len(base['y_true']) != len(comp['y_true']):
        raise ValueError(
            f"Fold count differs: {a_name} has {len(base['y_true'])}, "
            f"{b_name} has {len(comp['y_true'])}. Cannot pair."
        )

    have_ids = all(x is not None for x in base['row_ids'] + comp['row_ids'])
    for i in range(len(base['y_true'])):
        if have_ids:
            if not np.array_equal(base['row_ids'][i], comp['row_ids'][i]):
                raise ValueError(
                    f"Fold {i}: record ids differ between {a_name} and {b_name}. "
                    f"The two experiments were not run on the same patients in the "
                    f"same order, so a paired NRI is not defined. Re-run both with "
                    f"the same fold seed."
                )
        else:
            a, b = base['y_true'][i], comp['y_true'][i]
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(
                    f"Fold {i}: outcome vectors differ between {a_name} and {b_name} "
                    f"(n={a.shape} vs {b.shape}). The two experiments were not run on "
                    f"the same patients in the same order, so a paired NRI is not "
                    f"defined. Re-run both with the same fold seed."
                )
    return 'record_ids' if have_ids else 'outcomes'


# ============================================================================
# NRI
# ============================================================================

def nri_at_threshold(y_true: np.ndarray,
                     p_baseline: np.ndarray,
                     p_new: np.ndarray,
                     threshold: float) -> Dict[str, float]:
    """Two-category NRI at a single decision threshold.

    Positive values favour the new model. Counted from actual reclassification
    movements, then cross-checked against the sensitivity/specificity identity.
    Degenerate samples (no events or no non-events) return nan throughout, so
    that such bootstrap draws can be dropped rather than biasing the percentiles
    toward zero -- matching resample_metrics() in core_eval_functions.
    """
    y_true = np.asarray(y_true).astype(bool)
    high_base = np.asarray(p_baseline) >= threshold
    high_new = np.asarray(p_new) >= threshold

    events, non_events = y_true, ~y_true
    n_ev, n_nev = int(events.sum()), int(non_events.sum())
    if n_ev == 0 or n_nev == 0:
        return {k: np.nan for k in (
            'nri', 'event_component', 'non_event_component',
            'events_up', 'events_down', 'non_events_up', 'non_events_down',
            'n_events', 'n_non_events')}

    ev_up = int((events & ~high_base & high_new).sum())
    ev_down = int((events & high_base & ~high_new).sum())
    nev_up = int((non_events & ~high_base & high_new).sum())
    nev_down = int((non_events & high_base & ~high_new).sum())

    event_component = (ev_up - ev_down) / n_ev
    non_event_component = (nev_down - nev_up) / n_nev

    # Self-check: the components must equal the changes in sensitivity and
    # specificity. If this fires, the reclassification counting is wrong.
    d_sens = high_new[events].mean() - high_base[events].mean()
    d_spec = (~high_new[non_events]).mean() - (~high_base[non_events]).mean()
    assert np.isclose(event_component, d_sens, atol=1e-9), \
        f"event component {event_component} != delta sensitivity {d_sens}"
    assert np.isclose(non_event_component, d_spec, atol=1e-9), \
        f"non-event component {non_event_component} != delta specificity {d_spec}"

    return {
        'nri': event_component + non_event_component,
        'event_component': event_component,
        'non_event_component': non_event_component,
        'events_up': ev_up, 'events_down': ev_down,
        'non_events_up': nev_up, 'non_events_down': nev_down,
        'n_events': n_ev, 'n_non_events': n_nev,
    }


def bootstrap_nri_ci(y_true: np.ndarray,
                     p_baseline: np.ndarray,
                     p_new: np.ndarray,
                     threshold: float,
                     cluster_ids: Optional[np.ndarray] = None,
                     n_boot: int = 2000,
                     ci_level: float = 0.95,
                     seed: int = 0) -> Tuple[dict, dict]:
    """Percentile bootstrap CI for the pooled NRI, paired and cluster-aware.

    PAIRED: one set of resampled indices is applied to both models, preserving
    the within-patient correlation that makes the comparison informative.
    Resampling the two models independently would inflate the interval.

    CLUSTERED: when cluster_ids is given, draws clusters with replacement and
    keeps every row in a drawn cluster, exactly as bootstrap_ci() does. Required
    whenever one subject can contribute more than one row.

    Recalibration is NOT refitted inside the bootstrap: it was fitted on the
    training folds, which are not being resampled here, so the recalibrated
    probabilities are treated as fixed.
    """
    y_true = np.asarray(y_true)
    n = int(y_true.size)
    rng = np.random.default_rng(seed)

    groups = None
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.size != n:
            raise ValueError(
                f"cluster_ids has {cluster_ids.size} entries but there are {n} predictions")
        order = np.argsort(cluster_ids, kind='stable')
        _, starts = np.unique(cluster_ids[order], return_index=True)
        groups = np.split(order, starts[1:])

    draws: dict = {'nri': [], 'event_component': [], 'non_event_component': []}
    for _ in range(n_boot):
        idx = (rng.integers(0, n, n) if groups is None else
               np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))
        r = nri_at_threshold(y_true[idx], p_baseline[idx], p_new[idx], threshold)
        for k in draws:
            draws[k].append(r[k])

    lo_pct = (1 - ci_level) / 2 * 100
    hi_pct = 100 - lo_pct
    point = nri_at_threshold(y_true, p_baseline, p_new, threshold)

    results = {}
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        results[key] = {
            'point': point.get(key),
            'ci_lo': float(np.percentile(arr, lo_pct)),
            'ci_hi': float(np.percentile(arr, hi_pct)),
            'boot_sd': float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
            'n_valid_draws': int(arr.size),
        }

    meta = {
        'n_boot': n_boot, 'ci_level': ci_level,
        'resample_unit': 'cluster' if groups is not None else 'row',
        'n_rows': n,
        'n_clusters': int(len(groups)) if groups is not None else None,
    }
    if groups is not None:
        sizes = np.array([g.size for g in groups], dtype=float)
        meta['mean_cluster_size'] = float(sizes.mean())
        meta['max_cluster_size'] = int(sizes.max())
        meta['effective_cluster_size'] = float((sizes ** 2).sum() / sizes.sum())

    return results, meta


# ============================================================================
# TOP-LEVEL COMPARISON
# ============================================================================

def compare_experiments(baseline_dir: str | Path,
                        comparison_dirs: Sequence[str | Path],
                        threshold: float,
                        recalibrate: bool = True,
                        n_boot: int = 2000,
                        cluster_key: Optional[str] = None,
                        id_key: Optional[str] = None,
                        cluster_map: Optional[Dict[str, str]] = None,
                        ci_level: float = 0.95,
                        seed: int = 0,
                        labels: Optional[Sequence[str]] = None,
                        output_csv: Optional[str | Path] = None) -> pd.DataFrame:
    """Compute NRI for each comparison model against a common baseline.

    Args:
        baseline_dir:     experiment directory of the reference model.
        comparison_dirs:  experiment directories to compare against it.
        threshold:        decision threshold on the probability scale, e.g. 0.20.
        recalibrate:      fit logistic recalibration on each fold's training
                          predictions and apply to its test predictions. Must
                          match the setting used for the threshold metrics you
                          report alongside this table.
        n_boot:           bootstrap resamples for the pooled CI. 0 disables.
        cluster_key/id_key/cluster_map: as in ldh_eval, for cluster bootstrap.

    Returns:
        One row per comparison model: pooled NRI and components, bootstrap CI,
        reclassification counts, and fold-wise mean and SD.
    """
    comparison_dirs = list(comparison_dirs)
    if labels is None:
        labels = [resolve_input_dir(str(d)).name for d in comparison_dirs]
    if len(labels) != len(comparison_dirs):
        raise ValueError("labels must be the same length as comparison_dirs")

    want_clusters = bool(n_boot)
    base = load_experiment(baseline_dir, recalibrate=recalibrate,
                           cluster_key=cluster_key, id_key=id_key,
                           cluster_map=cluster_map, want_clusters=want_clusters)
    pooled_true = np.concatenate(base['y_true'])
    pooled_base = np.concatenate(base['y_prob'])

    if n_boot and base['cluster_ids'] is None:
        print("Warning: no cluster labels found — bootstrapping individual rows. If one "
              "subject can contribute more than one row, these intervals will be too "
              "narrow. See --cluster-key / --cluster-map.")

    rows = []
    for comp_dir, label in zip(comparison_dirs, labels):
        comp = load_experiment(comp_dir, recalibrate=recalibrate,
                               cluster_key=cluster_key, id_key=id_key,
                               cluster_map=cluster_map, want_clusters=False)
        how = _check_alignment(base, comp)

        pooled_comp = np.concatenate(comp['y_prob'])
        point = nri_at_threshold(pooled_true, pooled_base, pooled_comp, threshold)

        ci = {'ci_lo': np.nan, 'ci_hi': np.nan, 'boot_sd': np.nan}
        if n_boot:
            boot, meta = bootstrap_nri_ci(
                pooled_true, pooled_base, pooled_comp, threshold,
                cluster_ids=base['cluster_ids'], n_boot=n_boot,
                ci_level=ci_level, seed=seed)
            ci = boot.get('nri', ci)
            dropped = n_boot - ci.get('n_valid_draws', n_boot)
            if dropped:
                print(f"  note: {dropped}/{n_boot} resamples dropped as degenerate for {label}")

        fold_nris = np.asarray([
            nri_at_threshold(yt, pb, pc, threshold)['nri']
            for yt, pb, pc in zip(base['y_true'], base['y_prob'], comp['y_prob'])
        ], dtype=float)
        n_folds = int(np.sum(np.isfinite(fold_nris)))

        rows.append({
            'model': label,
            'baseline': base['name'],
            'threshold': threshold,
            'recalibrated': recalibrate,
            'alignment_checked_on': how,
            'nri_pooled': point['nri'],
            'nri_ci_lo': ci.get('ci_lo'),
            'nri_ci_hi': ci.get('ci_hi'),
            'nri_boot_sd': ci.get('boot_sd'),
            'event_component': point['event_component'],
            'non_event_component': point['non_event_component'],
            'events_up': point['events_up'],
            'events_down': point['events_down'],
            'non_events_up': point['non_events_up'],
            'non_events_down': point['non_events_down'],
            'n_events': point['n_events'],
            'n_non_events': point['n_non_events'],
            'nri_fold_mean': float(np.nanmean(fold_nris)),
            # Sample SD (ddof=1), matching evaluate_cross_validation: the folds
            # are a sample, not the whole population.
            'nri_fold_std': float(np.nanstd(fold_nris, ddof=1)) if n_folds > 1 else np.nan,
            'n_folds': n_folds,
        })

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✓ NRI results saved to {output_csv}")
    return df


def format_nri_table(df: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Manuscript-shaped view: one column per model, rows matching Table 5."""
    pct = int(round(ci_level * 100))
    out = {}
    for _, r in df.iterrows():
        if np.isfinite(r['nri_ci_lo']):
            pooled = (f"{r['nri_pooled']:+.3f} "
                      f"({r['nri_ci_lo']:+.3f} to {r['nri_ci_hi']:+.3f})")
        else:
            pooled = f"{r['nri_pooled']:+.3f}"
        out[r['model']] = {
            f'Pooled NRI ({pct}% CI)': pooled,
            'Event component': f"{r['event_component']:+.3f}",
            'Non-event component': f"{r['non_event_component']:+.3f}",
            'Mean fold NRI (SD)': f"{r['nri_fold_mean']:+.3f} ({r['nri_fold_std']:.3f})",
        }
    return pd.DataFrame(out)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Net Reclassification Improvement at a fixed decision threshold. '
                    'Uses the same recalibration map and cluster-bootstrap as ldh_eval.py.')
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Experiment directory of the reference model')
    parser.add_argument('--comparison_dirs', type=str, nargs='+', default=None,
                        help='One or more experiment directories to compare against the baseline. '
                             'Omit when using --ordering.')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Display names for the comparison models (default: directory names)')
    parser.add_argument('--ordering', type=str, default=None,
                        help='JSON file mapping experiment directory name -> label, as used by '
                             'ldh_eval --ordering. Supplies --comparison_dirs labels and order; '
                             'the baseline is excluded automatically.')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Decision threshold, e.g. 0.20')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Apply logistic recalibration (fit on train fold, applied to test '
                             'fold). Use this whenever the threshold metrics you report '
                             'alongside the NRI are recalibrated.')
    parser.add_argument('--bootstrap', type=int, default=2000, metavar='N',
                        help='Bootstrap resamples for the pooled CI (default 2000; 0 = off)')
    parser.add_argument('--cluster-key', type=str, default=None,
                        help='Key in the prediction JSON holding a per-row cluster label')
    parser.add_argument('--id-key', type=str, default=None,
                        help='Key in the prediction JSON holding a per-row record id, joined to '
                             'cluster labels via --cluster-map')
    parser.add_argument('--cluster-map', type=str, default=None,
                        help='CSV file mapping record id -> cluster id')
    parser.add_argument('--cluster-map-cols', type=str, default=None, metavar='ID,CLUSTER',
                        help='Column names in --cluster-map for record id and cluster id')
    parser.add_argument('--ci-level', type=float, default=0.95,
                        help='Confidence level for the bootstrap interval (default 0.95)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for bootstrap resampling (default 0)')
    parser.add_argument('--output_csv', type=str, default=None)

    args = parser.parse_args()

    if args.bootstrap < 0:
        parser.error('--bootstrap must be a non-negative number of resamples')
    if not 0 < args.ci_level < 1:
        parser.error('--ci-level must be strictly between 0 and 1 (e.g. 0.95)')
    if args.cluster_key and (args.cluster_map or args.id_key):
        parser.error('--cluster-key is mutually exclusive with --cluster-map/--id-key')
    if args.id_key and not args.cluster_map:
        parser.error('--id-key is only meaningful with --cluster-map')
    if args.cluster_map_cols and not args.cluster_map:
        parser.error('--cluster-map-cols requires --cluster-map')
    if (args.cluster_key or args.cluster_map or args.id_key) and not args.bootstrap:
        parser.error('cluster options only apply to bootstrap intervals; pass --bootstrap N')
    if args.ordering and args.labels:
        parser.error('--ordering and --labels are mutually exclusive')
    if bool(args.ordering) == bool(args.comparison_dirs):
        parser.error('pass exactly one of --comparison_dirs or --ordering')

    cluster_map = (load_cluster_map(args.cluster_map, args.cluster_map_cols)
                   if args.cluster_map else None)

    comparison_dirs, labels = args.comparison_dirs, args.labels
    if args.ordering:
        ordering = load_ordering(args.ordering)
        if not ordering:
            parser.error(f'{args.ordering} contained no experiments')
        base = resolve_input_dir(args.baseline_dir)
        names = [d for d, _ in ordering]
        if base.name not in names:
            parser.error(
                f"--baseline_dir '{base.name}' is not listed in {args.ordering}. "
                f"The ordering file names: {names}"
            )
        pairs = [(d, lab) for d, lab in ordering if d != base.name]
        if not pairs:
            parser.error(f'{args.ordering} lists only the baseline; nothing to compare against')
        missing = [d for d, _ in pairs if not (base.parent / d).is_dir()]
        if missing:
            parser.error(
                f"experiment director{'y' if len(missing) == 1 else 'ies'} named in "
                f"{args.ordering} not found under {base.parent}: {missing}"
            )
        comparison_dirs = [str(base.parent / d) for d, _ in pairs]
        labels = [lab for _, lab in pairs]
        print(f"Baseline: {ordering[names.index(base.name)][1]} ({base.name})")
        print(f"Comparing against {len(pairs)} experiment(s) from {args.ordering}")

    df = compare_experiments(
        args.baseline_dir, comparison_dirs, threshold=args.threshold,
        recalibrate=args.recalibrate, n_boot=args.bootstrap,
        cluster_key=args.cluster_key, id_key=args.id_key, cluster_map=cluster_map,
        ci_level=args.ci_level, seed=args.seed, labels=labels,
        output_csv=args.output_csv,
    )

    scale = 'RECALIBRATED' if args.recalibrate else 'RAW (uncalibrated)'
    print(f"\n=== NRI at threshold {args.threshold:.2f}, {scale} probabilities ===")
    print(f"Baseline: {resolve_input_dir(args.baseline_dir).name}")
    print(f"Alignment verified on: {df['alignment_checked_on'].iloc[0]}\n")
    print(format_nri_table(df, ci_level=args.ci_level).to_string())
    print()
