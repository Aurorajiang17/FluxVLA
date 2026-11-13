import json
from test.test_runners.test_process_sampling import make_runner

import numpy as np
import pytest

from fluxvla.engines.runners.serving.continuous_sampling import (
    ContinuousSamplingSession, continuous_sampling_config)
from fluxvla.engines.runners.serving.distributed_server import _predict_request
from fluxvla.engines.runners.serving.zmq_protocol import encode_frame

CASES = [('A:0', 1, 4), ('A:0', 2, 4), ('A:1', 1, 91), ('B:0', 1, 800)]


def collect(standard):
    runner, draws = make_runner(seed=7)
    outputs = []
    for index, (episode, sequence, seed) in enumerate(CASES):
        header = dict(
            request_id=str(index),
            episode_id=episode,
            sequence=sequence,
            policy_seed=seed)
        if not standard:
            header['sampling_mode'] = 'process_noise_stream_v1'
        body, _, extra = _predict_request(
            runner,
            header,
            encode_frame({'shape': (3, 8)}),
            accept_standard_process=standard)
        outputs.append(body['actions'].copy())
        if standard:
            assert set(body) == {'actions'}
            assert extra == {}
    assert runner.process_request_sequence == len(CASES)
    return outputs, draws


def test_standard_requests_preserve_process_draw_sequence():
    reference, expected_draws = collect(False)
    actual, actual_draws = collect(True)
    for left, right in zip(actual, reference):
        np.testing.assert_array_equal(left, right)
    assert actual_draws == expected_draws
    assert not np.array_equal(actual[0], actual[1])


def test_plain_request_still_rejected_without_opt_in():
    runner, _ = make_runner(seed=7)
    with pytest.raises(ValueError, match='do not match'):
        _predict_request(runner, dict(policy_seed=4), encode_frame({}))


@pytest.fixture
def session(tmp_path):
    value = ContinuousSamplingSession(tmp_path / 'audit.jsonl', 100000,
                                      'deploy', 7)
    value.bind_model('worker', 'instance')
    yield value
    value.close()


def header(**changes):
    return dict(
        request_id='request', episode_id='A', sequence=1,
        policy_seed=3) | changes


def test_session_audits_sequential_episodes(session, tmp_path):
    for index, (episode, sequence, seed) in enumerate(CASES):
        request = header(
            request_id=str(index),
            episode_id=episode,
            sequence=sequence,
            policy_seed=seed)
        assert session.accept((b'client', ), request, b'payload') == index + 1
        session.complete(str(index))
    records = [
        json.loads(line)
        for line in (tmp_path / 'audit.jsonl').read_text().splitlines()
    ]
    assert sum(r['event'] == 'accepted' for r in records) == 4
    assert sum(r['event'] == 'completed' for r in records) == 4


@pytest.mark.parametrize('change', [
    dict(sequence=True),
    dict(sequence=0),
    dict(sequence=2),
    dict(episode_id=''),
    dict(request_id=''),
    dict(policy_seed=True),
    dict(sampling_mode='legacy_episode'),
    dict(rng_state=b'x'),
    dict(rng_format='cpu')
])
def test_bad_request_seals_session(session, change):
    with pytest.raises((ValueError, RuntimeError)):
        session.accept((b'client', ), header(**change), b'')
    with pytest.raises(RuntimeError, match='sealed'):
        session.accept((b'client', ), header(), b'')


@pytest.mark.parametrize(
    'kind', ['duplicate', 'other_client', 'parallel', 'gap', 'old_episode'])
def test_request_ownership_and_order(session, kind):
    session.accept((b'client', ), header(), b'')
    if kind != 'parallel':
        session.complete('request')
    route = (b'other', ) if kind == 'other_client' else (b'client', )
    request = header(request_id='next', sequence=2)
    if kind == 'duplicate':
        request['request_id'] = 'request'
    if kind == 'gap':
        request['sequence'] = 3
    if kind == 'old_episode':
        session.accept(route, header(request_id='B', episode_id='B'), b'')
        session.complete('B')
        request = header(request_id='old')
    with pytest.raises((ValueError, RuntimeError)):
        session.accept(route, request, b'')


def test_model_replacement_seals_session(session):
    with pytest.raises(RuntimeError, match='replaced'):
        session.bind_model('worker', 'new-instance')


def test_audit_exclusive_and_capacity(tmp_path):
    path = tmp_path / 'audit'
    value = ContinuousSamplingSession(path, 5000, 'd', 0)
    with pytest.raises(FileExistsError):
        ContinuousSamplingSession(path, 5000, 'd', 0)
    value.bind_model('w', 'i')
    value._limit = value._bytes
    with pytest.raises(RuntimeError, match='capacity'):
        value.accept((b'c', ), header(), b'')
    with pytest.raises(RuntimeError, match='sealed'):
        value.accept((b'c', ), header(), b'')
    value.close()


def test_config_requires_single_process_and_audit(tmp_path):
    cfg = dict(
        runner=dict(sampling_mode='process_noise_stream_v1', process_seed=0),
        continuous_sampling=dict(enabled=True, audit_path=str(tmp_path / 'a')))
    assert continuous_sampling_config(cfg)['process_seed'] == 0
    with pytest.raises(ValueError, match='one model worker'):
        continuous_sampling_config(cfg, 2)
    cfg['runner']['process_resume'] = {'x': 1}
    with pytest.raises(ValueError, match='recovery'):
        continuous_sampling_config(cfg)
    assert continuous_sampling_config({}) is None


@pytest.mark.parametrize('field,value', [('request_id', None),
                                         ('episode_id', None),
                                         ('sequence', None),
                                         ('policy_seed', None),
                                         ('process_instance_id', 'caller'),
                                         ('process_sequence', 1),
                                         ('process_seed', 9),
                                         ('accept_standard_process', True)])
def test_missing_fields_and_caller_controls_seal(session, field, value):
    request = header()
    if value is None:
        request.pop(field)
    else:
        request[field] = value
    with pytest.raises((RuntimeError, ValueError)):
        session.accept((b'client', ), request, b'')
    with pytest.raises(RuntimeError, match='sealed'):
        session.accept((b'client', ), header(), b'')


def test_audit_write_failure_and_unfinished_shutdown(tmp_path, capsys):

    class BrokenFile:
        closed = False

        def write(self, data):
            raise OSError('synthetic disk error')

        def close(self):
            self.closed = True

    value = ContinuousSamplingSession(tmp_path / 'write-error', 10000, 'd', 7)
    value._file.close()
    value._file = BrokenFile()
    with pytest.raises(OSError, match='disk error'):
        value.bind_model('w', 'i')
    with pytest.raises(RuntimeError, match='sealed'):
        value.accept((b'c', ), header(), b'')
    assert 'continuous audit failed' in capsys.readouterr().err
    value.close()
    assert value._file.closed
    path = tmp_path / 'unfinished'
    value = ContinuousSamplingSession(path, 10000, 'd', 7)
    value.bind_model('w', 'i')
    value.accept((b'c', ), header(), b'')
    value.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]['event'] == 'failed'
    assert not any(r['event'] == 'completed' for r in records)


@pytest.mark.parametrize('changes', [{
    'enabled': 'yes'
}, {
    'max_audit_bytes': True
}, {
    'max_audit_bytes': 0
}, {
    'max_audit_bytes': 134217729
}, {
    'unknown': True
}, {
    'audit_path': None
}])
def test_invalid_service_config_rejected_before_model_load(tmp_path, changes):
    cfg = dict(
        runner=dict(sampling_mode='process_noise_stream_v1', process_seed=0),
        continuous_sampling=dict(enabled=True, audit_path=tmp_path / 'audit'))
    cfg['continuous_sampling'].update(changes)
    with pytest.raises((ValueError, TypeError)):
        continuous_sampling_config(cfg)
    assert not (tmp_path / 'audit').exists()


def test_close_audit_capacity_failure_is_sealed(tmp_path):
    value = ContinuousSamplingSession(tmp_path / 'close-error', 10000, 'd', 7)
    value._limit = value._bytes
    with pytest.raises(RuntimeError, match='capacity'):
        value.close()
    assert value._failed and value._file.closed
    with pytest.raises(RuntimeError, match='sealed'):
        value.accept((b'c', ), header(), b'')


def test_bind_error_seals_and_closes_broker_audit(tmp_path):
    import socket

    from fluxvla.engines.runners.serving.distributed_server import \
        FluxVLAInferenceServer
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        endpoint = f'tcp://127.0.0.1:{occupied.getsockname()[1]}'
        server = FluxVLAInferenceServer(
            frontend_bind=endpoint,
            backend_bind='tcp://127.0.0.1:1',
            continuous_sampling=dict(
                audit_path=tmp_path / 'bind-error',
                max_audit_bytes=10000,
                process_seed=7))
        with pytest.raises(Exception, match='Address already in use'):
            server.run()
    assert server._continuous._file.closed
    assert server._continuous._failed
    events = [
        json.loads(line)['event']
        for line in (tmp_path / 'bind-error').read_text().splitlines()
    ]
    assert events == ['started', 'failed']
