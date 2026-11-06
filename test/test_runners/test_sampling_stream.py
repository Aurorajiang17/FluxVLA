"""Cross-request sampling must advance independently of worker scheduling."""

import numpy as np
import pytest
import torch

from fluxvla.engines.runners.serving.serving_runner import FluxVLAServingRunner


class SamplingPolicy:
    supports_noise_generator = True

    def eval(self):
        return self

    def to(self, device, dtype=None):
        self.device = device
        return self

    def predict_action(self, shape=(3, 8), noise_generator=None, **kwargs):
        return torch.normal(
            0.0,
            1.0,
            size=shape,
            device=self.device,
            generator=noise_generator)


def make_runner(mode='episode_noise_stream_v1'):

    def dataset(observation):
        # The first noise starts after preprocessing has consumed RNG.
        torch.rand(3)
        return observation

    return FluxVLAServingRunner(
        SamplingPolicy(),
        dataset,
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True,
        **({
            'sampling_mode': mode
        } if mode != 'legacy_episode' else {}))


@pytest.mark.parametrize('width', [8, 14, 29])
def test_stream_matches_continuous_rng_and_survives_worker_transfer(width):
    workers = [make_runner(), make_runner()]
    torch.manual_seed(57)
    torch.rand(3)
    expected = [
        torch.normal(0.0, 1.0, size=(3, width)).numpy() for _ in range(12)
    ]
    state = fmt = None
    for index, reference in enumerate(expected):
        worker = workers[index % 2]
        args = ({'shape': (3, width)}, '', 57, index + 1, state, fmt)
        actual, _, next_state, next_fmt = (
            worker.predict_action_with_sampling_state(*args))
        replay, _, replay_state, replay_fmt = (
            workers[1 - index % 2].predict_action_with_sampling_state(*args))
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(replay, actual)
        assert next_state == replay_state
        assert next_fmt == replay_fmt
        # An unrelated episode cannot consume this episode's generator.
        worker.predict_action_with_sampling_state({}, '', 1057, 1)
        state, fmt = next_state, next_fmt


def test_legacy_repeats_noise():
    runner = make_runner('legacy_episode')
    first, _ = runner.predict_action({}, '', 57)
    second, _ = runner.predict_action({}, '', 57)
    np.testing.assert_array_equal(first, second)


def test_pi0_noise_generator_preserves_real_sampling_kernel():
    from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
    generator = torch.Generator(device='cpu').manual_seed(57)
    torch.manual_seed(57)
    for shape in [(1, 15, 32), (1, 8, 14), (2, 9, 29)]:
        expected = PI0FlowMatching.sample_noise(None, shape, 'cpu')
        actual = PI0FlowMatching.sample_noise(None, shape, 'cpu', generator)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('sequence,state,fmt', [
    (0, None, None),
    (True, None, None),
    (2, None, None),
    (1, b'old', 'old'),
    (2, b'bad', 'bad'),
])
def test_invalid_stream_state_is_rejected(sequence, state, fmt):
    runner = make_runner()
    with pytest.raises((TypeError, ValueError)):
        runner.predict_action_with_sampling_state({}, '', 57, sequence, state,
                                                  fmt)


def test_unsupported_model_cannot_silently_ignore_generator():
    model = SamplingPolicy()
    model.supports_noise_generator = False
    with pytest.raises(ValueError, match='noise_generator'):
        FluxVLAServingRunner(
            model,
            lambda x: x,
            device='cpu',
            model_outputs_environment_actions=True,
            sampling_mode='episode_noise_stream_v1')
