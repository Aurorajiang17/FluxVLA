"""Continuous default RNG sampling, with no model weights or GPU allocation."""
import random
from pathlib import Path
from test.test_runners.test_sampling_stream import SamplingPolicy

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines.runners.serving.distributed_server import (
    _predict_request, _sampling_context)
from fluxvla.engines.runners.serving.serving_runner import (
    FluxVLAServingRunner, configure_torch_runtime)
from fluxvla.engines.runners.serving.zmq_protocol import encode_frame


def make_runner(seed=7):

    class Policy(SamplingPolicy):

        def predict_action(self, **kwargs):
            assert 'seed' not in kwargs
            return super().predict_action(**kwargs)

    draws = []

    def dataset(observation):
        draws.append((random.random(), np.random.rand()))
        torch.rand(3)
        return observation

    runner = FluxVLAServingRunner(
        Policy(),
        dataset,
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True,
        sampling_mode='process_noise_stream_v1',
        process_seed=seed)
    return runner, draws


@pytest.mark.parametrize('width', [8, 29])
def test_default_stream_includes_preprocessing_and_ignores_episode_seeds(
        width):
    runner, draws = make_runner()
    actual = [
        runner.predict_action({'shape': (3, width)}, '', seed)[0]
        for seed in [4, 91, 4, 800, 17]
    ]
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected_draws = []
    for actions in actual:
        expected_draws.append((random.random(), np.random.rand()))
        torch.rand(3)
        expected = torch.normal(0., 1., size=(3, width)).numpy()
        np.testing.assert_array_equal(actions, expected)
    assert draws == expected_draws
    assert runner.process_request_sequence == 5


@pytest.mark.parametrize('seed', [None, True, -1, 2**32])
def test_process_seed_requires_an_explicit_supported_integer(seed):
    with pytest.raises(ValueError, match='process_seed'):
        make_runner(seed)


def test_model_failure_seals_the_stream():
    runner, _ = make_runner()

    def broken(**kwargs):
        torch.rand(3)
        raise RuntimeError('model failed')

    runner.vla.predict_action = broken
    with pytest.raises(RuntimeError, match='model failed'):
        runner.predict_action({}, '', 0)
    runner.vla = SamplingPolicy().to('cpu')
    with pytest.raises(RuntimeError, match='explicit recovery'):
        runner.predict_action({}, '', 1)


def test_dispatch_preserves_process_identity_across_new_episodes():
    runner, _ = make_runner()
    for index, episode in enumerate(['task-a:0', 'task-a:1', 'task-b:0'], 1):
        _, _, response = _predict_request(
            runner,
            dict(
                sampling_mode='process_noise_stream_v1',
                episode_id=episode,
                sequence=1,
                policy_seed=index), encode_frame({}))
        assert response['episode_id'] == episode
        assert response['process_sequence'] == index
        assert response['process_instance_id'] == runner.process_instance_id


@pytest.mark.parametrize(
    'change',
    [dict(rng_state=b'x'), dict(rng_format='cpu')])
def test_process_protocol_rejects_caller_rng_state(change):
    with pytest.raises(ValueError, match='caller RNG state'):
        _sampling_context(
            dict(
                sampling_mode='process_noise_stream_v1',
                episode_id='a',
                sequence=1,
                **change))


def test_process_recipe_preserves_the_entire_weight_and_compute_recipe():
    root = Path(__file__).resolve().parents[2] / 'configs/pi05'
    base = Config.fromfile(
        root / 'pi05_paligemma_robolab_droid_jointpos_inference.py').to_dict()
    process = Config.fromfile(
        root / 'pi05_paligemma_robolab_droid_jointpos_process_inference.py'
    ).to_dict()
    runner = process['themis']['server']['runner']
    assert runner.pop('process_seed') == 0
    assert runner['sampling_mode'] == 'process_noise_stream_v1'
    runner['sampling_mode'] = 'episode_noise_stream_v1'
    assert process['themis']['server'].pop('torch_runtime') == dict(
        num_threads=2,
        num_interop_threads=1,
        float32_matmul_precision='highest',
        allow_tf32_matmul=False,
        allow_tf32_cudnn=False)
    assert process == base


def test_explicit_runtime_controls_and_omitted_compatibility(monkeypatch):
    from types import SimpleNamespace

    import fluxvla.engines.runners.serving.serving_runner as module
    calls = []
    runtime = SimpleNamespace(
        set_num_threads=lambda n: calls.append(('threads', n)),
        get_num_interop_threads=lambda: 4,
        set_num_interop_threads=lambda n: calls.append(('interop', n)),
        set_float32_matmul_precision=lambda p: calls.append(('precision', p)),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setattr(module, 'torch', runtime)
    configure_torch_runtime(None)
    assert not calls and runtime.backends.cuda.matmul.allow_tf32
    configure_torch_runtime(
        dict(
            num_threads=2,
            num_interop_threads=1,
            float32_matmul_precision='highest',
            allow_tf32_matmul=False,
            allow_tf32_cudnn=False))
    assert calls == [('threads', 2), ('interop', 1), ('precision', 'highest')]
    assert not runtime.backends.cuda.matmul.allow_tf32
    assert not runtime.backends.cudnn.allow_tf32


@pytest.mark.parametrize('settings', [
    dict(num_threads=0),
    dict(num_interop_threads=True),
    dict(allow_tf32_matmul=1),
    dict(unrecognized=1)
])
def test_invalid_runtime_controls_do_not_mutate_torch(monkeypatch, settings):
    monkeypatch.setattr(torch, 'set_num_threads',
                        lambda *_: pytest.fail('mutated runtime'))
    with pytest.raises(ValueError):
        configure_torch_runtime(settings)


def test_process_checkpoint_restores_all_rngs_and_next_output(tmp_path):
    import hashlib
    import json
    runner, _ = make_runner()
    for _ in range(4):
        runner.predict_action({'shape': (3, 8)}, '', 0)
    runner.process_checkpoint_dir = str(tmp_path)
    before = json.dumps(runner.process_state(), sort_keys=True)
    runner.checkpoint_process_task('task-a')
    path = next(tmp_path.glob('*.json'))
    assert before == json.dumps(runner.process_state(), sort_keys=True)
    config = dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    expected = runner.predict_action({'shape': (3, 8)}, '', 0)[0]
    expected_state = json.dumps(runner.process_state(), sort_keys=True)
    restored, _ = make_runner()
    restored._restore_process_state(config)
    assert restored.process_request_sequence == 4
    actual = restored.predict_action({'shape': (3, 8)}, '', 999)[0]
    np.testing.assert_array_equal(actual, expected)
    assert expected_state == json.dumps(
        restored.process_state(), sort_keys=True)
    with pytest.raises(ValueError, match='hash'):
        restored._restore_process_state(dict(config, sha256='0' * 64))
    other, _ = make_runner(8)
    with pytest.raises(ValueError, match='configuration'):
        other._restore_process_state(config)


def test_checkpoint_task_survives_broker_context(tmp_path):
    runner, _ = make_runner()
    runner.process_checkpoint_dir = str(tmp_path)
    header = dict(
        sampling_mode='process_noise_stream_v1',
        episode_id='a',
        sequence=1,
        policy_seed=0,
        task_id='task-a')
    context = _sampling_context(header)
    assert context['task_id'] == 'task-a'
    _predict_request(runner, dict(context, policy_seed=0), encode_frame({}))
    _predict_request(runner, dict(context, policy_seed=0), encode_frame({}))
    import json
    states = list(tmp_path.glob('*.json'))
    assert len(states) == 1 and json.loads(
        states[0].read_text())['sequence'] == 0
