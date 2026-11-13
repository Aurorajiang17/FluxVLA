"""CPU ownership, reporting and audit checks for fixed continuous workers."""
import json

import pytest

from fluxvla.engines.runners.serving.continuous_sampling import \
    continuous_sampling_config
from fluxvla.engines.runners.serving.distributed_server import (
    FluxVLAInferenceServer, _ModelWorker)
from fluxvla.engines.runners.serving.fixed_worker_sampling import \
    FixedWorkerContinuousSession


def start_payload(counts=None, workers=2):
    counts = counts or {'A': 2, 'B': 2}
    ids = list(counts)
    return dict(
        tasks=[dict(task_id=t) for t in ids],
        task_ids=ids,
        total_tasks=len(ids),
        episodes_per_task_by_task=counts,
        total_episodes=sum(counts.values()),
        base_seed=4,
        parallel_workers=workers,
        strict_worker_assignment=True)


def start_session(tmp_path, workers=2, counts=None):
    session = FixedWorkerContinuousSession(workers, tmp_path / 'audit.jsonl',
                                           1000000, 'deployment', 7)
    for rank in reversed(range(workers)):
        session.bind_model(rank, f'model-worker-{rank}', f'instance-{rank}',
                           'cpu')
    report = session.prepare_report('run_start', 'run',
                                    start_payload(counts, workers))
    session.commit_report(report)
    return session


def request(ordinal, number=1, **changes):
    return dict(
        request_id=f'r{ordinal}-{number}',
        episode_id=f'run:{ordinal}',
        sequence=number,
        policy_seed=4 + ordinal - 1) | changes


def episode_report(session, kind, ordinal):
    episode = f'run:{ordinal}'
    payload = dict(session._jobs[episode])
    payload.pop('rank')
    if kind == 'episode_end':
        tickets = session._accepted_by_episode.get(episode, [])
        payload.update(
            prediction_metadata=[
                dict(
                    request_id=t.request_id,
                    episode_id=t.episode_id,
                    sequence=t.sequence) for t in tickets
            ],
            model_calls=len(tickets))
    session.commit_report(session.prepare_report(kind, 'run', payload))


def finish(session):
    session.commit_report(
        session.prepare_report(
            'run_end', 'run',
            dict(
                status='finished',
                completed_episodes=len(session._jobs),
                total_episodes=len(session._jobs))))
    session.close()


@pytest.mark.parametrize('workers', [1, 2, 3, 6, 8])
def test_parameterized_task_major_assignment(tmp_path, workers):
    session = start_session(tmp_path, workers, {'A': 3, 'B': 3})
    try:
        assert [j['rank'] for j in session._jobs.values()
                ] == [i % workers for i in range(6)]
        for ordinal in range(1, 7):
            episode_report(session, 'episode_start', ordinal)
            episode_report(session, 'episode_end', ordinal)
        finish(session)
        assert json.loads(
            (tmp_path /
             'audit.jsonl').read_text().splitlines()[-1])['requests'] == 0
    finally:
        session.close()


def test_reverse_arrival_and_completion_keep_original_tickets(tmp_path):
    session = start_session(tmp_path)
    try:
        second = session.accept((b'client-B', ), request(2), b'x')
        first = session.accept((b'client-A', ), request(1), b'y')
        assert (first.rank, second.rank) == (0, 1)
        session.complete(first.request_id, 'model-worker-0', 'instance-0', 1)
        session.complete(second.request_id, 'model-worker-1', 'instance-1', 1)
        for ordinal in (1, 2):
            episode_report(session, 'episode_start', ordinal)
            episode_report(session, 'episode_end', ordinal)
        for ordinal in (3, 4):
            episode_report(session, 'episode_start', ordinal)
            ticket = session.accept(
                (b'client-A' if ordinal == 3 else b'client-B', ),
                request(ordinal), b'z')
            session.complete(ticket.request_id, f'model-worker-{ticket.rank}',
                             f'instance-{ticket.rank}', 2)
            episode_report(session, 'episode_end', ordinal)
        finish(session)
        records = [
            json.loads(s)
            for s in (tmp_path / 'audit.jsonl').read_text().splitlines()
        ]
        completed = [r for r in records if r['event'] == 'completed']
        assert [r['global_sequence'] for r in completed] == [2, 1, 3, 4]
        assert [r['stream_sequence'] for r in completed] == [1, 1, 2, 2]
    finally:
        session.close()


@pytest.mark.parametrize('change', [
    dict(strict_worker_assignment=False),
    dict(parallel_workers=3),
    dict(parallel_workers=True),
    dict(resumed_from='old'),
    dict(restored_episodes=1),
    dict(task_ids=['B', 'A']),
    dict(tasks=[dict(task_id='A'), dict(task_id='A')]),
    dict(total_episodes=99),
    dict(total_tasks=True),
    dict(base_seed=True),
    dict(episodes_per_task_by_task={'A': 2}),
    dict(episodes_per_task_by_task={
        'A': 2,
        'B': False
    }),
])
def test_invalid_run_start_is_sealed(tmp_path, change):
    session = FixedWorkerContinuousSession(2, tmp_path / 'audit.jsonl', 100000,
                                           'deploy', 7)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            session.prepare_report('run_start', 'run',
                                   start_payload() | change)
        with pytest.raises(RuntimeError, match='sealed'):
            session.prepare_report('run_start', 'run', start_payload())
    finally:
        session.close()


@pytest.mark.parametrize('kind', [
    'other-client',
    'cross-rank',
    'parallel',
    'duplicate',
    'sequence-gap',
    'unknown-episode',
    'changed-seed',
    'caller-rng',
    'early-next-episode',
])
def test_invalid_prediction_does_not_become_accepted(tmp_path, kind):
    session = start_session(tmp_path)
    try:
        ticket = session.accept((b'A', ), request(1), b'x')
        if kind not in {'parallel', 'early-next-episode'}:
            session.complete(ticket.request_id, 'model-worker-0', 'instance-0',
                             1)
        bad, route = request(1, 2), (b'A', )
        if kind == 'other-client':
            route = (b'new-A', )
        elif kind == 'cross-rank':
            bad = request(2)
        elif kind == 'duplicate':
            bad['request_id'] = ticket.request_id
        elif kind == 'sequence-gap':
            bad['sequence'] = 3
        elif kind == 'unknown-episode':
            bad['episode_id'] = 'other:1'
        elif kind == 'changed-seed':
            bad['policy_seed'] = 999
        elif kind == 'caller-rng':
            bad['rng_state'] = b'x'
        elif kind == 'early-next-episode':
            bad = request(3)
        with pytest.raises((ValueError, RuntimeError)):
            session.accept(route, bad, b'x')
        assert len(session._tickets) == 1
    finally:
        session.close()


@pytest.mark.parametrize('change',
                         ['worker', 'instance', 'count', 'bool-count'])
def test_wrong_model_completion_is_sealed(tmp_path, change):
    session = start_session(tmp_path)
    ticket = session.accept((b'A', ), request(1), b'x')
    args = [ticket.request_id, 'model-worker-0', 'instance-0', 1]
    if change == 'worker':
        args[1] = 'model-worker-1'
    elif change == 'instance':
        args[2] = 'other'
    elif change == 'count':
        args[3] = 2
    else:
        args[3] = True
    try:
        with pytest.raises(ValueError):
            session.complete(*args)
        assert not session._completed
    finally:
        session.close()


@pytest.mark.parametrize('field', ['request_id', 'sequence', 'model_calls'])
def test_episode_end_checks_actual_prediction_receipts(tmp_path, field):
    session = start_session(tmp_path)
    try:
        ticket = session.accept((b'A', ), request(1), b'x')
        session.complete(ticket.request_id, 'model-worker-0', 'instance-0', 1)
        episode_report(session, 'episode_start', 1)
        payload = dict(session._jobs['run:1'])
        metadata = dict(
            request_id=ticket.request_id,
            episode_id=ticket.episode_id,
            sequence=ticket.sequence)
        payload.update(prediction_metadata=[metadata], model_calls=1)
        if field == 'model_calls':
            payload['model_calls'] = 2
        else:
            metadata[field] = 'wrong' if field == 'request_id' else 2
        with pytest.raises(ValueError, match='metadata'):
            session.prepare_report('episode_end', 'run', payload)
    finally:
        session.close()


def test_idle_unused_ranks_and_zero_predictions_can_finish(tmp_path):
    session = start_session(tmp_path, 3, {'A': 1})
    episode_report(session, 'episode_start', 1)
    episode_report(session, 'episode_end', 1)
    finish(session)
    closed = json.loads(
        (tmp_path / 'audit.jsonl').read_text().splitlines()[-1])
    assert closed['requests_by_rank'] == {'0': 0, '1': 0, '2': 0}


@pytest.mark.parametrize('kind', ['no-run-end', 'unfinished', 'cancelled'])
def test_incomplete_run_never_has_successful_closed(tmp_path, kind):
    session = start_session(tmp_path)
    if kind == 'unfinished':
        with pytest.raises(ValueError, match='unfinished'):
            session.prepare_report(
                'run_end', 'run',
                dict(
                    status='finished', total_episodes=4, completed_episodes=0))
    elif kind == 'cancelled':
        session.commit_report(
            session.prepare_report(
                'run_end', 'run',
                dict(
                    status='cancelled', total_episodes=4,
                    completed_episodes=0)))
    session.close()
    events = [
        json.loads(s)['event']
        for s in (tmp_path / 'audit.jsonl').read_text().splitlines()
    ]
    assert 'failed' in events and 'closed' not in events


def test_configuration_is_opt_in_and_uses_real_worker_count(tmp_path):
    cfg = dict(
        runner=dict(sampling_mode='process_noise_stream_v1', process_seed=0),
        evaluation_reporting=dict(enabled=True),
        continuous_sampling=dict(
            enabled=True,
            routing='fixed_worker',
            audit_path=str(tmp_path / 'audit.jsonl')))
    assert continuous_sampling_config(cfg, 6)['routing'] == 'fixed_worker'
    cfg['continuous_sampling'].pop('routing')
    with pytest.raises(ValueError, match='one model'):
        continuous_sampling_config(cfg, 2)
    assert 'routing' not in continuous_sampling_config(cfg, 1)
    cfg['continuous_sampling']['enabled'] = False
    assert continuous_sampling_config(cfg, 6) is None


def test_idle_heartbeat_limit_does_not_apply_to_busy_or_stopping_worker(
        tmp_path):
    server = FluxVLAInferenceServer(
        'tcp://127.0.0.1:1',
        'tcp://127.0.0.1:2',
        evaluation_report_writer=object(),
        expected_workers=1,
        continuous_sampling=dict(
            routing='fixed_worker',
            audit_path=tmp_path / 'audit.jsonl',
            max_audit_bytes=100000,
            process_seed=7))
    worker = _ModelWorker(
        b'w',
        'model-worker-0',
        'node',
        'cpu',
        is_ready=True,
        rank=0,
        last_seen=10)
    server._workers[b'w'] = worker
    try:
        worker.prediction = object()
        server._check_idle_workers(100)
        worker.prediction = None
        with pytest.raises(TimeoutError, match='heartbeat'):
            server._check_idle_workers(41)
        server._shutdown_started = True
        server._check_idle_workers(100)
    finally:
        server._continuous.close()


def test_duplicate_model_and_audit_capacity_fail_closed(tmp_path):
    session = start_session(tmp_path)
    with pytest.raises(RuntimeError, match='replaced'):
        session.bind_model(0, 'replacement', 'replacement', 'cpu')
    session.close()
    limited = FixedWorkerContinuousSession(1, tmp_path / 'limited.jsonl', 400,
                                           'deploy', 7)
    try:
        with pytest.raises(RuntimeError, match='capacity'):
            limited.bind_model(0, 'w' * 400, 'instance', 'cpu')
        assert limited._failed
    finally:
        limited.close()


@pytest.mark.parametrize('field,value', [('episode_index', 0.0),
                                         ('policy_seed', 4.0)])
def test_episode_numeric_fields_reject_float_coercion(tmp_path, field, value):
    session = start_session(tmp_path)
    payload = dict(session._jobs['run:1'])
    payload[field] = value
    try:
        with pytest.raises(ValueError, match='integer'):
            session.prepare_report('episode_start', 'run', payload)
    finally:
        session.close()


def test_heartbeat_cannot_replace_a_model(tmp_path):
    session = start_session(tmp_path)
    try:
        session.check_model_identity(0, 'model-worker-0', 'instance-0')
        with pytest.raises(ValueError, match='Heartbeat'):
            session.check_model_identity(0, 'model-worker-0', 'replacement')
    finally:
        session.close()


@pytest.mark.parametrize('failure', ['short-write', 'disk-error'])
def test_audit_io_failure_seals_before_dispatch(tmp_path, failure):
    session = start_session(tmp_path)
    stream = session._file

    class BrokenFile:

        @property
        def closed(self):
            return stream.closed

        def write(self, data):
            if failure == 'disk-error':
                raise OSError('synthetic disk failure')
            return len(data) - 1

        def close(self):
            stream.close()

    session._file = BrokenFile()
    try:
        with pytest.raises(OSError):
            session.accept((b'A', ), request(1), b'x')
        assert session._tickets == {}
        assert session._failed
    finally:
        session.close()
