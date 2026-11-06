"""RoboLab leaderboard statistics without simulator or SciPy dependencies."""

from __future__ import annotations
import math
from collections import defaultdict

import numpy as np

METRIC_NAMES = ('ee_sparc', 'joint_sparc_mean', 'ee_isj', 'joint_isj',
                'ee_path_length', 'joint_rmse_mean', 'ee_speed_max',
                'ee_speed_mean')


def clopper_pearson(successes, total):
    """Exact central 95% binomial interval, returned on the [0, 1] scale."""
    if not 0 <= successes <= total:
        raise ValueError('Invalid binomial counts')
    if not total:
        return None, None
    indices = np.arange(total + 1, dtype=np.float64)
    log_coeff = np.array([
        math.lgamma(total + 1) - math.lgamma(i + 1) -
        math.lgamma(total - i + 1) for i in range(total + 1)
    ])

    def bound(lower):
        lo, hi = 0.0, 1.0
        for _ in range(64):
            p = (lo + hi) / 2
            mass = np.exp(log_coeff + indices * math.log(p) +
                          (total - indices) * math.log1p(-p))
            probability = (
                mass[successes:].sum() if lower else mass[:successes +
                                                          1].sum())
            increase = probability < 0.025 if lower else probability > 0.025
            if increase:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (bound(True) if successes else 0.0,
            bound(False) if successes < total else 1.0)


def score_bootstrap(task_scores, samples=10000, seed=0):
    """Two-stage task/episode percentile CI for the task macro mean.

    Uses a dedicated PCG64 stream. Repeated selections of the same task have
    independent within-task episode samples. Official resampling parameters
    are unpublished; these parameters are disclosed in the report.
    """
    if not task_scores:
        return None, None
    values = [
        np.asarray(task_scores[key], dtype=float)
        for key in sorted(task_scores)
    ]
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = rng.integers(len(values), size=(samples, len(values)))
    means = np.empty(selected.shape)
    for index, scores in enumerate(values):
        positions = np.where(selected == index)
        draws = rng.integers(
            len(scores), size=(len(positions[0]), len(scores)))
        means[positions] = scores[draws].mean(axis=1)
    return tuple(np.percentile(means.mean(axis=1) * 100, [2.5, 97.5]))


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _complete_metrics(episode):
    info = episode.get('info', {})
    metrics = info.get('metrics', {})
    status = info.get('metric_status', {})
    return info.get('metrics_complete') is True and all(
        name in metrics and (_finite(metrics[name]) or
                             (metrics[name] is None and status.get(name) in
                              {'native_undefined', 'too_few_samples'}))
        for name in METRIC_NAMES)


def summarize_metrics(episodes, planned, intervals=True):
    """Include success and failure trajectories, exposing missing coverage."""
    scores = defaultdict(list)
    for episode in episodes:
        score = episode.get('score')
        if _finite(score) and 0 <= score <= 1:
            scores[episode['task_id']].append(score)
    flat = [score for values in scores.values() for score in values]
    sr_ci = (
        clopper_pearson(sum(
            e['success'] for e in episodes), len(episodes)) if intervals else
        (None, None))
    score_ci = score_bootstrap(scores) if intervals else (None, None)
    result = dict(
        success_rate_ci95_low_pct=None if sr_ci[0] is None else sr_ci[0] * 100,
        success_rate_ci95_high_pct=None if sr_ci[1] is None else sr_ci[1] *
        100,
        score_mean_pct=float(np.mean(flat) * 100) if flat else None,
        score_task_macro_pct=(float(
            np.mean([np.mean(v)
                     for v in scores.values()]) * 100) if scores else None),
        score_valid_episodes=len(flat),
        score_ci95_low_pct=score_ci[0],
        score_ci95_high_pct=score_ci[1],
        metrics_complete=(len(episodes) == planned and len(flat) == planned
                          and all(_complete_metrics(e) for e in episodes)))
    for name in METRIC_NAMES:
        values = [
            e.get('info', {}).get('metrics', {}).get(name) for e in episodes
        ]
        values = [value for value in values if _finite(value)]
        result[name + '_valid_episodes'] = len(values)
        result[name] = float(np.mean(values)) if values else None
        if name == 'ee_speed_mean':
            result['ee_speed_cm_s'] = float(np.mean(values) *
                                            100) if values else None
            result['ee_speed_sd_cm_s'] = float(np.std(values, ddof=0) *
                                               100) if values else None
    return result


STATISTICAL_PROTOCOL = dict(
    version='robolab_leaderboard_v1',
    confidence=0.95,
    success_rate_ci='Clopper-Pearson',
    score_ci='two-stage task/episode percentile bootstrap; task macro mean',
    bootstrap_samples=10000,
    bootstrap_seed=0,
    bootstrap_rng='PCG64',
    numpy_version=np.__version__,
    official_bootstrap_parameters_verified=False,
    ee_speed_units='cm/s',
    ee_speed_sd_ddof=0,
    trajectory_aggregation=('equal weight per valid episode; '
                            'successes and failures'),
    score_aggregation='episode mean; task macro mean also reported',
    ee_speed_excludes_inference_time=True)
