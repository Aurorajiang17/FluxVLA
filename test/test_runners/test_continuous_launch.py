"""CPU checks of the production torchrun entry and configuration call chain."""
import importlib.util
from pathlib import Path

import pytest

from fluxvla.engines.runners.serving import distributed_server as serving


def _config(tmp_path, routing='fixed_worker', reporting=True):
    path = tmp_path / 'config.py'
    audit = tmp_path / 'audit.jsonl'
    path.write_text(
        'eval=dict(dataset={})\nthemis=dict(server=dict('
        'runner=dict(sampling_mode="process_noise_stream_v1",'
        'process_seed=7),evaluation_reporting=dict('
        f'enabled={reporting!r}),continuous_sampling=dict('
        f'enabled=True,routing={routing!r},audit_path={str(audit)!r})))\n')
    return path, audit


def _launch(config, rank, workers, **kwargs):
    return serving.launch_server_task(
        config_path=str(config),
        ckpt_path='unused',
        device=f'cuda:{rank}',
        frontend_bind='tcp://127.0.0.1:15555',
        backend_bind='tcp://127.0.0.1:15556',
        backend_endpoint='tcp://127.0.0.1:15556',
        max_pending_requests=128,
        request_timeout_s=120,
        startup_timeout_s=900,
        rank=rank,
        world_size=workers,
        **kwargs)


@pytest.mark.parametrize('workers', [1, 2, 3, 6, 8])
def test_all_launch_ranks_share_deployment_and_actual_count(
        tmp_path, monkeypatch, workers):
    config, audit = _config(tmp_path)
    monkeypatch.setenv('FLUXVLA_DEPLOYMENT_ID', 'parent-deployment')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', str(workers))
    model_calls, broker_calls = [], []

    class Process:
        exitcode = 0

        def __init__(self, target, kwargs, name):
            broker_calls.append(kwargs)

        def start(self):
            audit.write_text('broker-owned\n')

        def join(self):
            pass

        def is_alive(self):
            return False

    Context = type('Context', (), {'Process': Process})

    monkeypatch.setattr(serving.multiprocessing, 'get_context',
                        lambda _: Context())
    monkeypatch.setattr(serving, 'run_model_worker',
                        lambda **kw: model_calls.append(kw))
    for rank in range(workers):
        _launch(config, rank, workers)
    assert len(broker_calls) == 1
    assert broker_calls[0]['expected_workers'] == workers
    assert [c['worker_id'] for c in model_calls
            ] == [f'model-worker-{r}' for r in range(workers)]
    assert {c['deployment_id'] for c in model_calls} == {'parent-deployment'}
    assert {c['expected_workers'] for c in model_calls} == {workers}
    assert audit.read_text() == 'broker-owned\n'


@pytest.mark.parametrize('case', [
    'deployment', 'nodes', 'reporting', 'single-multiple', 'rank', 'rank-bool'
])
def test_launch_rejects_invalid_configuration_before_model_load(
        tmp_path, monkeypatch, case):
    config, _ = _config(
        tmp_path,
        routing=('single_client'
                 if case == 'single-multiple' else 'fixed_worker'),
        reporting=case != 'reporting')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1' if case == 'nodes' else '2')
    monkeypatch.setenv('FLUXVLA_DEPLOYMENT_ID',
                       '' if case == 'deployment' else 'run')
    calls = []
    monkeypatch.setattr(serving, 'run_model_worker',
                        lambda **kw: calls.append(kw))
    rank = True if case == 'rank-bool' else 2 if case == 'rank' else 1
    with pytest.raises((ValueError, TypeError)):
        _launch(config, rank, 2)
    assert calls == []


def test_server_process_passes_worker_count_to_configuration(
        tmp_path, monkeypatch):
    config, audit = _config(tmp_path)
    monkeypatch.setattr(serving, '_configure_child_lifetime', lambda _: None)
    writer = object()
    monkeypatch.setattr(serving, 'build_evaluation_report_writer_from_config',
                        lambda *a, **kw: writer)
    constructed = []

    def run(server):
        constructed.append(server)
        server._continuous.close()

    monkeypatch.setattr(serving.FluxVLAInferenceServer, 'run', run)
    serving._server_process_main(
        config_path=str(config),
        ckpt_path='unused',
        cfg_options={},
        frontend_bind='tcp://127.0.0.1:15555',
        backend_bind='tcp://127.0.0.1:15556',
        expected_workers=3,
        max_pending_requests=128,
        request_timeout_s=120,
        startup_timeout_s=900,
        deployment_id='run',
        parent_pid=1)
    assert constructed[0].expected_workers == 3
    assert constructed[0].evaluation_report_writer is writer
    assert audit.exists()
    with pytest.raises(FileExistsError):
        serving._server_process_main(
            config_path=str(config),
            ckpt_path='unused',
            cfg_options={},
            frontend_bind='tcp://127.0.0.1:15555',
            backend_bind='tcp://127.0.0.1:15556',
            expected_workers=3,
            max_pending_requests=128,
            request_timeout_s=120,
            startup_timeout_s=900,
            deployment_id='run',
            parent_pid=1)


@pytest.mark.parametrize('ids', ['1,3,5', 'GPU-example-a,GPU-example-b'])
def test_public_cli_uses_local_visible_rank_and_actual_world_size(
        tmp_path, monkeypatch, ids):
    count = len(ids.split(','))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ids)
    monkeypatch.setenv('WORLD_SIZE', str(count))
    monkeypatch.setenv('LOCAL_WORLD_SIZE', str(count))
    monkeypatch.setenv('RANK', str(count - 1))
    monkeypatch.setenv('LOCAL_RANK', str(count - 1))
    config, _ = _config(tmp_path)
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'synthetic unused checkpoint')
    calls = []
    monkeypatch.setattr(serving, 'launch_server_task',
                        lambda **kw: calls.append(kw))
    script = Path(__file__).parents[2] / 'scripts/distributed_zmq_server.py'
    spec = importlib.util.spec_from_file_location('public_zmq_cli', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        '--config',
        str(config), '--ckpt-path',
        str(checkpoint), '--backend-endpoint', 'tcp://127.0.0.1:15556'
    ]) == 0
    assert calls[0]['device'] == f'cuda:{count - 1}'
    assert calls[0]['world_size'] == count
    assert calls[0]['rank'] == count - 1
    assert calls[0]['backend_endpoint'] == 'tcp://127.0.0.1:15556'
