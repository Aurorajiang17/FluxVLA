"""RoboLab report contracts using synthetic events, no simulator or model."""

import csv
import json
from pathlib import Path

import pytest
from mmengine.config import Config

from fluxvla.engines.runners.serving import evaluation_report_writer as reports
from fluxvla.engines.runners.serving.distributed_server import \
    build_evaluation_report_writer_from_config


class _Run:

    def __init__(self, tmp_path, kind='robolab', counts=None):
        self.counts = counts or {'task-b': 1, 'task-a': 3}
        task_list = ['OpenSingleDoor'] if kind == 'robocasa' else None
        if kind == 'robocasa':
            self.counts = {'0': 1}
        self.writer = reports.FluxVLAEvaluationReportWriter(
            result_root=tmp_path,
            config_path=tmp_path / 'config.py',
            ckpt_path=tmp_path / 'model.pt',
            eval_config=dict(
                model_family='pi05',
                num_trials_per_task=1,
                task_suite_name=kind,
                task_list=task_list),
            report_kind=kind,
            logger=lambda message: None)
        self.sequence = 1
        self.completed = 0
        self.total = sum(self.counts.values())
        tasks = []
        for index, task_id in enumerate(self.counts):
            metadata = dict(benchmark=kind, task_name=task_id)
            if task_list:
                metadata['env_name'] = task_list[index]
            tasks.append(
                dict(
                    task_id=task_id,
                    task_index=index if task_list else 17 - index * 13,
                    description=f'Do {task_id}',
                    metadata=metadata))
        self.start_payload = dict(
            schema_version='1.0',
            run_name='cpu-report',
            base_seed=7,
            episodes_per_task=1,
            episodes_per_task_by_task=self.counts,
            max_episode_steps=4500,
            execute_horizon=None,
            total_tasks=len(tasks),
            total_episodes=self.total,
            tasks=tasks)

    def send(self, event, payload, accepted=True, sequence=None):
        response = self.writer.process_event(
            event,
            request_id=f'request-{self.sequence}',
            run_session_id='session',
            sequence=self.sequence if sequence is None else sequence,
            payload=payload)
        assert response['accepted'] is accepted, response
        if accepted:
            self.sequence += 1
        return response

    def start(self):
        self.send('run_start', self.start_payload)
        return self

    def episode_start(self, task_id, index):
        payload = dict(
            task_id=task_id,
            episode_index=index,
            episode_id=f'session:{task_id}:{index}',
            policy_seed=7 + index,
            environment_seed=11 + index,
            environment_case_index=index,
            started_at='2026-10-10T00:00:00+00:00')
        self.send('episode_start', payload)
        return payload

    def end_payload(self, start, success=False, frozen=True):
        info = dict(
            benchmark='robolab',
            task_id=start['task_id'],
            episode_index=start['episode_index'],
            environment_seed=start['environment_seed'],
            environment_case_index=start['environment_case_index'],
            simulator_seed=0,
            environment_instance=2,
            instance_episode=start['episode_index'],
            native_output_dir='/simulator-only/worker-2/instance-2',
            native_frozen=frozen,
            native_result=dict(
                env_id=0,
                success=success if frozen else None,
                step=8 if frozen else None))
        return dict(
            **start,
            success=success,
            episode_return=float(success),
            steps=8,
            duration_s=1.5,
            termination_reason='truncated' if frozen else 'max_steps',
            model_calls=1,
            info=info,
            prediction_metadata=[{
                'request_id': 'prediction-1'
            }],
            completed_episodes=self.completed + 1,
            total_episodes=self.total)

    def episode(self, task_id, index, success=False, frozen=True):
        start = self.episode_start(task_id, index)
        end = self.end_payload(start, success, frozen)
        self.send('episode_end', end)
        self.completed += 1
        return end

    def finish(self, status='finished', **extra):
        payload = dict(
            status=status,
            finished_at='2026-10-10T00:00:04+00:00',
            duration_s=4.0,
            completed_episodes=self.completed,
            total_episodes=self.total,
            **extra)
        response = self.send('run_end', payload)
        directory = Path(response['summary_path']).parent
        summary = json.loads((directory / 'summary.json').read_text())
        return summary, directory, response


def test_episode_csv_identifies_native_batch_demos(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 3}).start()
    for index in range(3):
        start = run.episode_start('task-a', index)
        end = run.end_payload(start)
        end['info']['native_hdf5_path'] = '/native/run_0.hdf5'
        if index < 2:
            end['info'].update(
                native_demo_key=f'demo_{index}', native_batch_lane=index)
        run.send('episode_end', end)
        run.completed += 1
    _, directory, _ = run.finish()
    with (directory / 'episodes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['native_hdf5_path'], r['native_demo_key'])
            for r in rows[:2]] == [('/native/run_0.hdf5', 'demo_0'),
                                   ('/native/run_0.hdf5', 'demo_1')]
    assert [r['native_batch_lane'] for r in rows] == ['0', '1', '']
    assert rows[2]['native_demo_key'] == ''


def test_robolab_weighted_counts_files_and_provenance(tmp_path, monkeypatch):

    def no_upload(*args, **kwargs):
        pytest.fail('RoboLab reporting must remain local')

    monkeypatch.setattr(reports, 'maybe_report_summary_to_feishu', no_upload)
    run = _Run(tmp_path).start()
    # Complete in a different order from the runtime manifest.
    run.episode('task-a', 0, True)
    run.episode('task-b', 0, True)
    run.episode('task-a', 1)
    last = run.episode('task-a', 2)
    summary, directory, response = run.finish()
    overall = summary['overall']
    assert overall['planned_episodes'] == 4
    assert overall['completed_episodes'] == overall['native_episodes'] == 4
    assert overall['successes'] == overall['failures'] == 2
    assert overall['success_rate_pct'] == 50.0
    assert overall['unscored_episodes'] == overall['missing_episodes'] == 0
    assert overall['plan_complete'] and overall['results_complete']
    assert summary['report_complete'] is True
    assert summary['full_suite'] is None
    assert summary['native_artifacts_verified'] is False
    assert response['reported_to_feishu'] is False
    assert 'RoboLab' in response['report_reason']
    assert summary['task_stats']['task-a'][
        'success_rate_pct'] == pytest.approx(100 / 3)
    task = json.loads((directory / 'robolab/task4_results.json').read_text())
    episode = task['episodes'][-1]
    assert episode['info'] == last['info']
    assert episode['policy_seed'] == 9
    assert episode['environment_seed'] == 13
    assert episode['info']['simulator_seed'] == 0
    assert episode['prediction_metadata'] == last['prediction_metadata']
    journal = [
        json.loads(line)
        for line in (directory / 'events.jsonl').read_text().splitlines()
    ]
    assert [e['sequence'] for e in journal] == list(range(1, 11))
    assert journal[-2]['payload']['info'] == last['info']
    with (directory / 'summary.csv').open(newline='') as stream:
        row = next(csv.DictReader(stream))
    assert float(row['success_rate_pct']) == overall['success_rate_pct']
    assert int(row['native_episodes']) == 4
    with (directory / 'task_success_rates.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {row['task_id'] for row in rows} == {'task-a', 'task-b'}
    text = (directory / 'summary.txt').read_text()
    assert '50.00%' in text and 'finished' in text
    assert '4/4' in text
    assert (directory / 'failed_tasks.txt').read_text() == ''
    assert not (tmp_path / 'robolab').exists()  # No legacy GPU mirror.


@pytest.mark.parametrize('status', ['finished', 'failed', 'interrupted'])
def test_robolab_partial_and_zero_results(tmp_path, status):
    run = _Run(tmp_path).start()
    if status != 'finished':
        run.episode_start('task-a', 0)
    summary, directory, _ = run.finish(status, error={'message': 'stopped'})
    assert summary['status'] == status
    assert summary['error'] == {'message': 'stopped'}
    assert not summary['report_complete']
    assert not summary['overall']['plan_complete']
    assert summary['overall']['missing_episodes'] == 4
    assert summary['overall']['success_rate_pct'] is None
    assert len(summary['task_stats']) == 2
    assert 'N/A' in (directory / 'summary.txt').read_text()
    with (directory / 'summary.csv').open(newline='') as stream:
        assert next(csv.DictReader(stream))['success_rate_pct'] == ''


def test_robolab_unfinished_native_episode_is_unscored(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 2}).start()
    run.episode('task-a', 0, True)
    run.episode('task-a', 1, frozen=False)
    summary, directory, _ = run.finish()
    overall = summary['overall']
    assert overall['plan_complete']
    assert not overall['results_complete'] and not summary['report_complete']
    assert overall['native_episodes'] == overall['unscored_episodes'] == 1
    assert overall['success_rate_pct'] == 100.0
    assert overall['failures'] == 0
    task = json.loads((directory / 'robolab/task17_results.json').read_text())
    assert task['episodes'][1]['unscored_reason'] == 'native_not_frozen'
    progress = json.loads((directory / 'rank_progress/rank0.json').read_text())
    assert progress['native_episodes'] == 1
    assert progress['success_rate_pct'] == 100.0
    assert (directory / 'failed_tasks.txt').read_text() == '17\n'


@pytest.mark.parametrize('change, message', [
    ({
        'native_result': {
            'success': True,
            'step': 8
        }
    }, 'success'),
    ({
        'native_result': {
            'success': False,
            'step': None
        }
    }, 'step'),
    ({
        'native_result': {
            'success': 'false',
            'step': 8
        }
    }, 'success'),
    ({
        'native_frozen': 'yes'
    }, 'native_frozen'),
    ({
        'native_frozen': False
    }, 'native_frozen'),
    ({
        'task_id': 'different'
    }, 'task_id'),
    ({
        'episode_index': 99
    }, 'episode_index'),
    ({
        'environment_seed': 99
    }, 'environment_seed'),
    ({
        'benchmark': 'robodojo'
    }, 'benchmark'),
])
def test_robolab_rejects_conflicting_native_evidence(tmp_path, change,
                                                     message):
    run = _Run(tmp_path, counts={'task-a': 1}).start()
    start = run.episode_start('task-a', 0)
    payload = run.end_payload(start)
    payload['info'].update(change)
    response = run.send('episode_end', payload, accepted=False)
    assert message in response['error']
    # Rejection neither consumes the sequence nor increments the count.
    run.send('episode_end', run.end_payload(start))
    run.completed += 1
    summary, _, _ = run.finish()
    assert summary['overall']['completed_episodes'] == 1


def test_robolab_missing_evidence_does_not_create_a_failure(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 1}).start()
    start = run.episode_start('task-a', 0)
    end = run.end_payload(start)
    end['info'] = {}
    run.send('episode_end', end)
    run.completed += 1
    summary, _, _ = run.finish()
    assert summary['overall']['failures'] == 0
    assert summary['overall']['unscored_episodes'] == 1
    assert summary['overall']['success_rate_pct'] is None


def test_robolab_sequence_identity_and_duplicate_guards(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 1}).start()
    start = run.episode_start('task-a', 0)
    end = run.end_payload(start)
    run.send('episode_end', end, accepted=False, sequence=99)
    bad_seed = {**end, 'policy_seed': 100}
    assert 'policy_seed' in run.send(
        'episode_end', bad_seed, accepted=False)['error']
    run.send('episode_end', end)
    run.completed += 1
    run.send('episode_end', end, accepted=False)
    run.send('episode_start', start, accepted=False)
    summary, _, _ = run.finish()
    assert summary['overall']['completed_episodes'] == 1


def test_robolab_manifest_and_active_run_guards(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 1})
    bad = {**run.start_payload, 'total_episodes': 2}
    run.send('run_start', bad, accepted=False)
    run.start_payload['tasks'][0]['metadata']['benchmark'] = 'robodojo'
    run.send('run_start', run.start_payload, accepted=False)
    run.start_payload['tasks'][0]['metadata']['benchmark'] = 'robolab'
    run.start()
    run.episode_start('task-a', 0)
    end = dict(
        status='finished',
        finished_at='2026-10-10T00:00:04+00:00',
        duration_s=4.,
        completed_episodes=0,
        total_episodes=1)
    assert 'active' in run.send('run_end', end, accepted=False)['error']
    run.finish('interrupted')


def test_robolab_retry_and_interruption_keep_only_accepted_outcomes(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 2}).start()
    start = run.episode_start('task-a', 0)
    retry = {**start, 'environment_case_index': 8}
    run.send('episode_start', retry)
    stale_end = run.end_payload(start, True)
    run.send('episode_end', stale_end, accepted=False)
    run.send('episode_end', run.end_payload(retry, True))
    run.completed += 1
    run.episode_start('task-a', 1)
    summary, directory, _ = run.finish(
        'failed', error={
            'type': 'TimeoutError',
            'message': 'model timeout'
        })
    assert summary['overall']['completed_episodes'] == 1
    assert summary['overall']['missing_episodes'] == 1
    assert summary['overall']['success_rate_pct'] == 100.0
    assert not summary['report_complete']
    task = json.loads((directory / 'robolab/task17_results.json').read_text())
    assert len(task['episodes']) == 1
    assert task['episodes'][0]['environment_case_index'] == 8


def test_robolab_unverified_success_is_rejected(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 1}).start()
    start = run.episode_start('task-a', 0)
    payload = run.end_payload(start, True, frozen=False)
    assert 'success' in run.send(
        'episode_end', payload, accepted=False)['error']


def test_robolab_finished_label_does_not_hide_missing_quota(tmp_path):
    run = _Run(tmp_path, counts={'task-a': 2}).start()
    run.episode('task-a', 0)
    summary, directory, _ = run.finish()
    assert summary['status'] == 'finished'
    assert summary['overall']['missing_episodes'] == 1
    assert summary['overall']['failures'] == 1
    assert not summary['report_complete']
    assert 'Partial results' in (directory / 'summary.txt').read_text()


def test_robolab_recipe_builds_reporter_without_loading_model(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config_path = root / (
        'configs/pi05/pi05_paligemma_robolab_droid_jointpos_inference.py')
    cfg = Config.fromfile(str(config_path))
    assert cfg.themis.server.dataset_section == 'eval'
    assert cfg.themis.server.evaluation_reporting.enabled
    checkpoint = tmp_path / 'checkpoints/model.safetensors'
    checkpoint.parent.mkdir()
    checkpoint.touch()  # An empty file proves no weights are loaded.
    cfg.themis.server.evaluation_reporting.result_output_dir = 'reports'
    writer = build_evaluation_report_writer_from_config(
        cfg, str(checkpoint), str(config_path))
    assert writer.report_kind == 'robolab'
    assert writer.task_suite_name == 'robolab'
    assert writer.expected_task_indexes is None
    assert writer.result_root == tmp_path / 'reports'
    cfg.themis.server.evaluation_reporting.enabled = False
    assert build_evaluation_report_writer_from_config(cfg, str(checkpoint),
                                                      str(config_path)) is None


@pytest.mark.parametrize('kind', ['libero', 'robodojo', 'robocasa'])
def test_existing_benchmark_summary_paths_remain_available(tmp_path, kind):
    run = _Run(tmp_path, kind=kind, counts={'pick': 1}).start()
    task_id = next(iter(run.counts))
    start = run.episode_start(task_id, 0)
    end = run.end_payload(start, True)
    # These benchmarks do not carry RoboLab native evidence.
    end['info'] = {}
    run.send('episode_end', end)
    run.completed += 1
    summary, directory, _ = run.finish()
    assert (directory / 'summary.csv').is_file()
    assert (directory / 'summary.txt').is_file()
    assert 'report_complete' not in summary  # No new RoboLab schema leakage.


def test_terminal_metrics_reach_episode_csv_and_leaderboard_summary(tmp_path):
    from fluxvla.engines.runners.serving.robolab_statistics import METRIC_NAMES
    run = _Run(tmp_path, counts={'task': 1}).start()
    start = run.episode_start('task', 0)
    end = run.end_payload(start, False)
    end['score'] = 0.5
    end['info'].update(
        metrics={
            **dict.fromkeys(METRIC_NAMES, 0.0), 'ee_speed_mean': 0.03,
            'ee_sparc': -8.0
        },
        metrics_complete=True,
        control_dt_s=1 / 15,
        simulation_duration_s=8 / 15,
        metrics_compute_s=0.02,
        native_hdf5_path='/simulator/run_0.hdf5')
    run.send('episode_end', end)
    run.completed += 1
    summary, directory, _ = run.finish()
    assert summary['metrics_report_complete']
    assert summary['overall']['score_mean_pct'] == 50
    assert summary['overall']['ee_speed_cm_s'] == 3
    assert summary['overall']['ee_speed_sd_cm_s'] == 0
    assert summary['overall']['ee_sparc'] == -8
    with (directory / 'episodes.csv').open() as file:
        row = next(csv.DictReader(file))
    assert row['score'] == '0.5'
    assert row['success'] == 'False'
    assert row['native_hdf5_path'] == '/simulator/run_0.hdf5'
    assert summary['statistical_protocol']['bootstrap_seed'] == 0
