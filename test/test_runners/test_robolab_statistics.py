import numpy as np
import pytest

from fluxvla.engines.runners.serving.robolab_statistics import (
    METRIC_NAMES, clopper_pearson, score_bootstrap, summarize_metrics)


@pytest.mark.parametrize('k,n,expected', [
    (0, 0, (None, None)),
    (0, 1, (0, 0.975)),
    (1, 1, (0.025, 1)),
    (336, 1200, (0.25474402972593552, 0.306338798843704)),
])
def test_clopper_pearson_matches_published_interval(k, n, expected):
    result = clopper_pearson(k, n)
    assert result == pytest.approx(expected, abs=1e-11)


def test_bootstrap_uses_task_clusters_and_owns_rng():
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    # Pooled episode bootstrapping would nearly erase the small task.
    scores = {'a': [0.0] * 100, 'b': [1.0]}
    first = score_bootstrap(scores)
    assert first == (0, 100)
    assert np.random.random() == expected_next
    assert score_bootstrap(dict(reversed(list(scores.items())))) == first
    assert score_bootstrap({'a': [0.5]}) == (50, 50)


def test_metrics_include_failed_progress_population_sd_and_missing_counts():
    defaults = dict.fromkeys(METRIC_NAMES, 0.0)
    episodes = [
        dict(
            task_id='a',
            success=False,
            score=0.5,
            info=dict(
                metrics={
                    **defaults, 'ee_speed_mean': 0.01,
                    'ee_sparc': -6
                },
                metrics_complete=True)),
        dict(
            task_id='b',
            success=True,
            score=1,
            info=dict(
                metrics={
                    **defaults, 'ee_speed_mean': 0.03,
                    'ee_sparc': None
                },
                metric_status={'ee_sparc': 'native_undefined'},
                metrics_complete=True))
    ]
    result = summarize_metrics(episodes, 2)
    assert result['score_mean_pct'] == 75
    assert result['ee_speed_cm_s'] == pytest.approx(2)
    assert result['ee_speed_sd_cm_s'] == pytest.approx(1)
    assert result['ee_sparc'] == -6
    assert result['ee_sparc_valid_episodes'] == 1
    assert result['metrics_complete']
    assert not summarize_metrics(episodes, 3)['metrics_complete']
    episodes[0]['score'] = None
    result = summarize_metrics(episodes, 2)
    assert result['score_mean_pct'] == 100
    assert result['score_valid_episodes'] == 1
    assert not result['metrics_complete']
