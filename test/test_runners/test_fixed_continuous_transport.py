"""Real clients, broker, report writer and independent CPU model processes."""
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from test.test_runners.test_continuous_transport import _client  # noqa: F401
from test.test_runners.test_continuous_transport import _endpoint, _stop
from test.test_runners.test_continuous_transport import \
    themis as _themis_fixture
from test.test_runners.test_process_sampling import make_runner

import numpy as np
import pytest

from fluxvla.engines.runners.serving import distributed_server as serving
from fluxvla.engines.runners.serving.evaluation_report_writer import \
    FluxVLAEvaluationReportWriter

themis = _themis_fixture


def _fixed_service(tmp_path, fault=None, workers=2):
    ctx = multiprocessing.get_context('fork')
    frontend, backend = _endpoint(), _endpoint()
    audit = tmp_path / 'audit.jsonl'
    config = tmp_path / 'service.py'
    config.write_text(
        'themis=dict(server=dict(runner=dict('
        'sampling_mode="process_noise_stream_v1",process_seed=7),'
        'evaluation_reporting=dict(enabled=True),'
        'continuous_sampling=dict(enabled=True,routing="fixed_worker",'
        f'audit_path={str(audit)!r})))\n')
    entered = [ctx.Event(), ctx.Event()]
    release = [ctx.Event(), ctx.Event()]
    calls = ctx.Queue()

    def broker():
        writer = FluxVLAEvaluationReportWriter(
            result_root=tmp_path / 'reports',
            config_path=config,
            ckpt_path=tmp_path / 'model.pt',
            eval_config=dict(
                model_family='pi05',
                num_trials_per_task=1,
                task_suite_name='robolab'),
            report_kind='robolab',
            logger=lambda msg: None)
        serving.FluxVLAInferenceServer(
            frontend_bind=frontend,
            backend_bind=backend,
            expected_workers=workers,
            deployment_id='fixed-integration',
            evaluation_report_writer=writer,
            request_timeout_s=0.2 if fault == 'timeout' else 8,
            startup_timeout_s=8,
            continuous_sampling=dict(
                routing='fixed_worker',
                audit_path=audit,
                max_audit_bytes=1000000,
                process_seed=7)).run()

    def model_worker(rank):
        runner, _ = make_runner(seed=7)
        serving.build_serving_runner_from_config = lambda *a, **kw: runner
        original = serving._predict_request
        number = 0

        def traced(current, header, payload, **kwargs):
            nonlocal number
            number += 1
            calls.put((rank, dict(header)))
            if number == 1:
                entered[rank].set()
                if not release[rank].wait(8):
                    raise TimeoutError('CPU test barrier timed out')
            if rank == 0 and fault == 'model-error':
                raise RuntimeError('synthetic fixed model error')
            output, elapsed, response = original(current, header, payload,
                                                 **kwargs)
            if rank == 0 and fault == 'wrong-instance':
                current.process_instance_id = 'wrong-instance'
            if rank == 0 and fault == 'wrong-count':
                current.process_request_sequence += 1
            if rank == 0 and fault == 'invalid-actions':
                output['actions'] = np.full((3, 8), np.nan)
            return output, elapsed, response

        serving._predict_request = traced
        serving.run_model_worker(
            str(config),
            'unused',
            backend,
            f'model-worker-{rank}',
            'cpu-test',
            'fixed-integration',
            'cpu',
            expected_workers=workers)

    processes = [ctx.Process(target=broker)] + [
        ctx.Process(target=model_worker, args=(r, ))
        for r in reversed(range(workers))
    ]
    for process in processes:
        process.start()
    return frontend, processes, entered, release, calls, audit


def _reporter(endpoint):
    from fluxthemis.reporters.fluxvla_reporter import FluxVLAEvaluationReporter
    return FluxVLAEvaluationReporter(
        transport=dict(
            endpoint=endpoint, connect_timeout_s=8, request_timeout_s=8))


def _begin(reporter, counts=None, workers=2):
    counts = counts or {'A': 2, 'B': 2}
    reporter.setup()
    reporter.report_evaluation(
        'run_start',
        dict(
            schema_version='1.0',
            run_name='fixed-cpu',
            base_seed=7,
            episodes_per_task=2,
            episodes_per_task_by_task=counts,
            total_tasks=len(counts),
            total_episodes=sum(counts.values()),
            max_episode_steps=4500,
            execute_horizon=None,
            tasks=[
                dict(
                    task_id=t,
                    task_index=i,
                    description=t,
                    metadata=dict(benchmark='robolab', task_name=t))
                for i, t in enumerate(counts)
            ],
            task_ids=list(counts),
            parallel_workers=workers,
            strict_worker_assignment=True))


def _start(reporter, task, index, ordinal):
    return dict(
        task_id=task,
        episode_index=index,
        episode_id=f'{reporter.run_session_id}:{ordinal}',
        policy_seed=7 + ordinal - 1,
        environment_seed=7 + ordinal - 1,
        environment_case_index=index,
        started_at='2026-10-10T00:00:00+00:00')


def _end(start, chunks, completed, total=4, workers=2):
    rank = (int(start['episode_id'].rsplit(':', 1)[1]) - 1) % workers
    return dict(
        **start,
        success=False,
        episode_return=0.,
        steps=8,
        duration_s=1.,
        termination_reason='truncated',
        model_calls=len(chunks),
        prediction_metadata=[c.metadata for c in chunks],
        completed_episodes=completed,
        total_episodes=total,
        info=dict(
            benchmark='robolab',
            task_id=start['task_id'],
            episode_index=start['episode_index'],
            environment_seed=start['environment_seed'],
            environment_case_index=start['environment_case_index'],
            simulator_seed=0,
            environment_instance=rank,
            instance_episode=0,
            native_output_dir='/synthetic/native',
            native_frozen=True,
            native_result=dict(env_id=0, success=False, step=8)))


def _predict(client):
    return client.predict_action(
        dict(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            state=np.zeros(8, dtype=np.float32)),
        dict(task_description='task'))


@pytest.mark.parametrize('arrival', [(1, 0), (0, 1)])
def test_fixed_real_parallel_models_and_interleaved_reports(
        themis, tmp_path, arrival):
    cls, task_spec = themis
    endpoint, processes, entered, release, calls, audit = _fixed_service(
        tmp_path)
    reporter = _reporter(endpoint)
    clients = [_client(cls, endpoint), _client(cls, endpoint)]
    chunks = {0: [], 1: []}
    request_headers = {0: [], 1: []}
    try:
        _begin(reporter)
        for client in clients:
            client.setup()
        starts = [_start(reporter, 'A', rank, rank + 1) for rank in (0, 1)]
        for rank, client in enumerate(clients):
            client.reset(
                task_spec(task_id='A'), rank, starts[rank]['policy_seed'])
            client.set_episode_id(starts[rank]['episode_id'])
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for rank in arrival:
                futures[rank] = executor.submit(_predict, clients[rank])
                assert entered[rank].wait(5), 'Rank dispatch was blocked'
            # Both real model processes entered before either was released.
            # Prediction arrived before episode_start on a separate connection.
            for rank in reversed(arrival):
                reporter.report_evaluation('episode_start', starts[rank])
            for rank in reversed(arrival):
                release[rank].set()
                chunks[rank].append(futures[rank].result(timeout=5))
        for rank in arrival:
            chunks[rank].append(_predict(clients[rank]))
        completed = 0
        for rank in reversed(arrival):
            completed += 1
            reporter.report_evaluation(
                'episode_end', _end(starts[rank], chunks[rank], completed))
        # One rank finishes its entire quota before the other starts task B.
        for rank in arrival:
            start = _start(reporter, 'B', rank, rank + 3)
            reporter.report_evaluation('episode_start', start)
            clients[rank].reset(
                task_spec(task_id='B'), rank, start['policy_seed'])
            clients[rank].set_episode_id(start['episode_id'])
            chunk = _predict(clients[rank])
            chunks[rank].append(chunk)
            completed += 1
            reporter.report_evaluation('episode_end',
                                       _end(start, [chunk], completed))
        reporter.report_evaluation(
            'run_end',
            dict(
                status='finished',
                finished_at='2026-10-10T00:00:04+00:00',
                duration_s=4.,
                completed_episodes=4,
                total_episodes=4))
        summary_path = reporter.evaluation_report_run_dir
        for _ in range(6):
            rank, header = calls.get(timeout=2)
            request_headers[rank].append(header)
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        for event in release:
            event.set()
        for client in clients:
            client.close()
        reporter.close()
        _stop(processes)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    accepted = [r for r in records if r['event'] == 'accepted']
    finished = [r for r in records if r['event'] == 'completed']
    assert [r['rank'] for r in accepted[:2]] == list(arrival)
    assert [r['rank'] for r in finished[:2]] == list(reversed(arrival))
    assert {r['request_id']
            for r in accepted} == {r['request_id']
                                   for r in finished}
    assert records[-1]['event'] == 'closed'
    assert records[-1]['requests_by_rank'] == {'0': 3, '1': 3}
    assert len({r['process_instance_id'] for r in finished}) == 2
    for rank in (0, 1):
        rank_records = [r for r in finished if r['rank'] == rank]
        assert [r['process_sequence'] for r in rank_records] == [1, 2, 3]
        assert len({r['process_instance_id'] for r in rank_records}) == 1
        reference, _ = make_runner(seed=7)
        for chunk, header in zip(chunks[rank], request_headers[rank]):
            expected, _ = reference.predict_action({}, '',
                                                   header['policy_seed'])
            np.testing.assert_array_equal(chunk.actions, expected)
            assert all(
                key not in header and key not in chunk.metadata
                for key in ('rng_state', 'sampling_mode',
                            'process_instance_id', 'process_sequence'))
    np.testing.assert_array_equal(chunks[0][0].actions, chunks[1][0].actions)
    assert not np.array_equal(chunks[0][0].actions, chunks[0][-1].actions)
    from pathlib import Path
    summary = json.loads((Path(summary_path) / 'summary.json').read_text())
    assert summary['report_complete'] is True
    assert summary['status'] == 'finished'


@pytest.mark.parametrize('fault', [
    'model-error', 'timeout', 'writer-rejection', 'missing-receipt',
    'cancelled', 'interrupted', 'interrupted-report', 'wrong-instance',
    'wrong-count', 'invalid-actions'
])
def test_fixed_real_failures_seal_whole_service(themis, tmp_path, fault):
    cls, task_spec = themis
    endpoint, processes, entered, release, calls, audit = _fixed_service(
        tmp_path, fault)
    reporter = _reporter(endpoint)
    client = _client(cls, endpoint)
    try:
        _begin(reporter)
        client.setup()
        start = _start(reporter, 'A', 0, 1)
        client.reset(task_spec(task_id='A'), 0, start['policy_seed'])
        client.set_episode_id(start['episode_id'])
        if fault == 'writer-rejection':
            # Valid session state; authoritative writer rejects the gap.
            reporter._sequence += 1
            with pytest.raises(RuntimeError):
                reporter.report_evaluation('episode_start', start)
        elif fault == 'cancelled':
            # The authoritative writer rejects this unsupported status.
            with pytest.raises(RuntimeError):
                reporter.report_evaluation(
                    'run_end',
                    dict(
                        status='cancelled',
                        finished_at='2026-10-10T00:00:04+00:00',
                        duration_s=4.,
                        completed_episodes=0,
                        total_episodes=4))
        elif fault == 'interrupted-report':
            reporter.report_evaluation(
                'run_end',
                dict(
                    status='interrupted',
                    finished_at='2026-10-10T00:00:04+00:00',
                    duration_s=4.,
                    completed_episodes=0,
                    total_episodes=4))
        elif fault == 'interrupted':
            # A signal before run_end must produce incomplete audit, even idle.
            processes[0].terminate()
        else:
            if fault != 'timeout':
                release[0].set()
            if fault in {
                    'timeout', 'model-error', 'wrong-instance', 'wrong-count',
                    'invalid-actions'
            }:
                with pytest.raises((RuntimeError, TimeoutError)):
                    _predict(client)
            else:
                chunk = _predict(client)
                reporter.report_evaluation('episode_start', start)
                end = _end(start, [chunk], 1)
                end['prediction_metadata'] = []
                with pytest.raises(RuntimeError):
                    reporter.report_evaluation('episode_end', end)
        processes[0].join(5)
        assert not processes[0].is_alive()
        if fault not in {'interrupted', 'interrupted-report'}:
            assert processes[0].exitcode != 0
    finally:
        for event in release:
            event.set()
        client.close()
        reporter.close()
        _stop(processes)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records[-1]['event'] == 'failed'
    assert sum(r['event'] == 'accepted' for r in records) <= 1
    assert not any(r['event'] == 'closed' for r in records)


def test_fixed_one_worker_matches_continuous_reference(themis, tmp_path):
    cls, task_spec = themis
    endpoint, processes, entered, release, calls, audit = _fixed_service(
        tmp_path, workers=1)
    reporter = _reporter(endpoint)
    client = _client(cls, endpoint)
    chunks = []
    try:
        _begin(reporter, counts={'A': 1, 'B': 1}, workers=1)
        client.setup()
        release[0].set()
        for ordinal, task in enumerate(('A', 'B'), 1):
            start = _start(reporter, task, 0, ordinal)
            client.reset(task_spec(task_id=task), 0, start['policy_seed'])
            client.set_episode_id(start['episode_id'])
            chunk = _predict(client)
            chunks.append(chunk)
            reporter.report_evaluation('episode_start', start)
            reporter.report_evaluation(
                'episode_end',
                _end(start, [chunk], ordinal, total=2, workers=1))
        reporter.report_evaluation(
            'run_end',
            dict(
                status='finished',
                finished_at='2026-10-10T00:00:04+00:00',
                duration_s=4.,
                completed_episodes=2,
                total_episodes=2))
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        release[0].set()
        client.close()
        reporter.close()
        _stop(processes)
    reference, _ = make_runner(seed=7)
    for seed, chunk in zip((7, 8), chunks):
        actions, _ = reference.predict_action({}, '', seed)
        np.testing.assert_array_equal(chunk.actions, actions)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records[-1]['event'] == 'closed'
    assert records[-1]['requests_by_rank'] == {'0': 2}
