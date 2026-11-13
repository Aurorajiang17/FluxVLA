"""CPU integration against the actual optional FluxThemis checkout.

Run with --cpu-model-tests and FLUXTHEMIS_SOURCE pointing to that checkout.
Only the model factory is replaced; broker, worker and client are real modules.
"""
import json
import multiprocessing
import os
import socket
import time
from pathlib import Path
from test.test_runners.test_process_sampling import make_runner

import numpy as np
import pytest

from fluxvla.engines.runners.serving import distributed_server as serving
from fluxvla.engines.runners.serving.zmq_protocol import (empty_payload,
                                                          encode_frame,
                                                          encode_header)


def _endpoint():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'tcp://127.0.0.1:{sock.getsockname()[1]}'


@pytest.fixture
def themis(monkeypatch):
    source = os.environ.get('FLUXTHEMIS_SOURCE')
    if not source:
        pytest.skip('Set FLUXTHEMIS_SOURCE for cross-repository integration')
    assert (Path(source) / 'fluxthemis').is_dir()
    monkeypatch.syspath_prepend(source)
    from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
    from fluxthemis.structures import TaskSpec
    return FluxVLAPolicyClient, TaskSpec


def _start_service(tmp_path,
                   fault,
                   deployment_id='integration',
                   frontend=None):
    ctx = multiprocessing.get_context('fork')
    frontend, backend = frontend or _endpoint(), _endpoint()
    audit = tmp_path / 'audit.jsonl'
    config = tmp_path / 'service.py'
    config.write_text(
        'themis = dict(server=dict(\n'
        'runner=dict(sampling_mode="process_noise_stream_v1", '
        'process_seed=7), continuous_sampling=dict(enabled=True, '
        f'audit_path={str(audit)!r})))\n')
    calls = ctx.Queue()

    def broker():
        serving.FluxVLAInferenceServer(
            frontend_bind=frontend,
            backend_bind=backend,
            expected_workers=1,
            deployment_id=deployment_id,
            request_timeout_s=0.1 if fault == 'timeout' else 3,
            startup_timeout_s=5,
            continuous_sampling=dict(
                audit_path=audit, max_audit_bytes=100000,
                process_seed=7)).run()

    def model_worker():
        runner, _ = make_runner(seed=7)
        if fault == 'model_error':

            def broken(**kwargs):
                raise RuntimeError('synthetic model failure')

            runner.vla.predict_action = broken
        elif fault == 'timeout':
            original = runner.vla.predict_action

            def slow(**kwargs):
                time.sleep(2)
                return original(**kwargs)

            runner.vla.predict_action = slow
        serving.build_serving_runner_from_config = lambda *a, **k: runner
        original_predict = serving._predict_request

        def traced(current, header, payload, **kwargs):
            calls.put(dict(header))
            return original_predict(current, header, payload, **kwargs)

        serving._predict_request = traced
        serving.run_model_worker(
            str(config), 'unused', backend, 'worker', 'cpu-test',
            deployment_id, 'cpu')

    processes = [ctx.Process(target=broker), ctx.Process(target=model_worker)]
    for process in processes:
        process.start()
    return frontend, processes, calls, audit


def _stop(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(2)
        assert not process.is_alive()


def _client(cls, endpoint):
    return cls(
        dict(
            endpoint=endpoint,
            image_keys=['rgb'],
            state_keys=['state'],
            connect_timeout_s=5,
            request_timeout_s=3))


def test_real_client_broker_worker_preserves_continuous_stream(
        themis, tmp_path):
    cls, task_spec = themis
    endpoint, processes, calls, audit = _start_service(tmp_path, None)
    client = _client(cls, endpoint)
    observation = dict(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32))
    chunks, requests = [], []
    try:
        client.setup()
        # Neither readiness nor reporting enters the model worker.
        client.transport.request(
            encode_header('status', request_id='status'), empty_payload())
        report, _ = client.transport.request(
            encode_header(
                'report_event',
                request_id='report',
                deployment_id='integration'), encode_frame({}))
        assert not report['ok']  # Reporting disabled in this CPU fixture.
        for task, episode, count, seed in [('A', 0, 2, 4), ('A', 1, 1, 91),
                                           ('B', 0, 1, 800)]:
            client.reset(
                task_spec(task_id=task, description=task), episode, seed)
            for _ in range(count):
                chunks.append(
                    client.predict_action(observation,
                                          {'task_description': 'task'}))
                requests.append(calls.get(timeout=2))
        assert calls.empty()
        assert [r['request_id'] for r in requests
                ] == [c.metadata['request_id'] for c in chunks]
        assert [r['sequence'] for r in requests] == [1, 2, 1, 1]
        assert [r['episode_id']
                for r in requests] == ['A:0', 'A:0', 'A:1', 'B:0']
        assert all('sampling_mode' not in r and 'rng_state' not in r
                   for r in requests)
        reference, _ = make_runner(seed=7)
        for chunk, request in zip(chunks, requests):
            expected, _ = reference.predict_action({}, '',
                                                   request['policy_seed'])
            np.testing.assert_array_equal(chunk.actions, expected)
    finally:
        client.close()
        _stop(processes)
    assert processes[0].exitcode == 0
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    accepted = [r for r in records if r['event'] == 'accepted']
    completed = [r for r in records if r['event'] == 'completed']
    assert [r['request_id']
            for r in accepted] == [r['request_id'] for r in completed]
    assert [r['request_id']
            for r in completed] == [c.metadata['request_id'] for c in chunks]
    assert sum(r['event'] == 'model_ready' for r in records) == 1
    assert records[-1]['event'] == 'closed'


@pytest.mark.parametrize('fault',
                         ['duplicate', 'model_error', 'timeout', 'deployment'])
def test_transport_failure_seals_service(themis, tmp_path, fault):
    cls, task_spec = themis
    endpoint, processes, calls, audit = _start_service(tmp_path, fault)
    client = _client(cls, endpoint)
    observation = dict(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), state=np.zeros(8))
    try:
        client.setup()
        client.reset(task_spec(task_id='A', description='A'), 0, 4)
        if fault in ('duplicate', 'deployment'):
            chunk = client.predict_action(observation,
                                          {'task_description': 'task'})
            calls.get(timeout=2)
            client._prediction_sequence = 1
            header, payload = client._build_request(
                chunk.metadata['request_id'], observation,
                {'task_description': 'task'})
            if fault == 'deployment':
                from fluxvla.engines.runners.serving.zmq_protocol import \
                    decode_header
                values = decode_header(header)
                values['deployment_id'] = 'previous-deployment'
                header = encode_frame(values)
            response, _ = client.transport.request(header, payload)
            assert not response['ok']
        else:
            with pytest.raises((RuntimeError, TimeoutError)):
                client.predict_action(observation,
                                      {'task_description': 'task'})
            calls.get(timeout=2)
        processes[0].join(5)
        assert processes[0].exitcode != 0
        assert not processes[0].is_alive()
        assert calls.empty()  # No hidden re-evaluation.
    finally:
        client.close()
        _stop(processes)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records[-1]['event'] == 'failed'
    assert sum(r['event'] == 'accepted' for r in records) == 1


def test_new_service_instance_cannot_resume_existing_client(themis, tmp_path):
    cls, task_spec = themis
    first = tmp_path / 'first'
    second = tmp_path / 'replacement'
    first.mkdir()
    second.mkdir()
    endpoint, old_processes, old_calls, old_audit = _start_service(first, None)
    client = _client(cls, endpoint)
    new_processes = []
    observation = dict(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), state=np.zeros(8))
    try:
        client.setup()
        client.reset(task_spec('A'), 0, 4)
        client.predict_action(observation, dict(task_description='A'))
        old_calls.get(timeout=2)
        _stop(old_processes)
        _, new_processes, new_calls, new_audit = _start_service(
            second, None, deployment_id='replacement', frontend=endpoint)
        probe = _client(cls, endpoint)
        try:
            probe.setup()  # Status only; it does not own the prediction route.
            assert probe.transport.server_deployment_id == 'replacement'
        finally:
            probe.close()
        client.reset(task_spec('A'), 1, 91)
        with pytest.raises(RuntimeError, match='deployment_id mismatch'):
            client.predict_action(observation, dict(task_description='A'))
        new_processes[0].join(5)
        assert new_processes[0].exitcode != 0
        assert not new_processes[0].is_alive()
        assert new_calls.empty()
    finally:
        client.close()
        _stop(old_processes + new_processes)
    old = [json.loads(line) for line in old_audit.read_text().splitlines()]
    new = [json.loads(line) for line in new_audit.read_text().splitlines()]
    assert old[1]['process_instance_id'] != new[1]['process_instance_id']
    assert old[0]['deployment_id'] != new[0]['deployment_id']
    assert new[-1]['event'] == 'failed'
    assert not any(r['event'] == 'accepted' for r in new)
