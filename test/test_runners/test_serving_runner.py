import numpy as np
import pytest
import torch

from fluxvla.engines.runners.serving.serving_runner import FluxVLAServingRunner


class _Policy:

    def __init__(self, actions=None):
        self.actions = actions
        self.seeds = []
        self.device = None
        self.dtype = None

    def eval(self):
        return self

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def predict_action(self, **batch):
        self.seeds.append(batch['seed'])
        return torch.zeros(2, 3) if self.actions is None else self.actions


def test_serving_runner_receives_episode_seed():
    model = _Policy()
    runner = FluxVLAServingRunner(
        model,
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=True,
        model_outputs_environment_actions=True,
    )

    actions, _ = runner.predict_action({}, '', policy_seed=17)

    assert actions.shape == (2, 3)
    assert model.seeds == [17]
    assert model.dtype == torch.bfloat16
    assert model.device == torch.device('cpu')


@pytest.mark.parametrize('value', [
    np.ones((2, 8), dtype=bool),
    np.full((2, 8), '0.25'),
    np.full((2, 8), 0.25 + 1j),
    np.full((2, 8), 0.25, dtype=object),
    [[True] + [0.25] * 7] * 2,
    [np.zeros(8), np.ones(8, dtype=bool)],
    torch.ones(2, 8, dtype=torch.bool),
    torch.ones(2, 8, dtype=torch.complex64),
])
@pytest.mark.parametrize('denormalized', [False, True])
def test_serving_rejects_non_real_actions_before_conversion(
        value, denormalized):
    runner = FluxVLAServingRunner(
        _Policy(value),
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=not denormalized,
        denormalize_action=lambda payload: payload['action'])
    with pytest.raises(TypeError, match='real numbers'):
        runner.predict_action({}, '', policy_seed=0)


def test_serving_validates_denormalizer_output():
    runner = FluxVLAServingRunner(
        _Policy(),
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=False,
        denormalize_action=lambda payload: [[True, 0.0, 0.0]])
    with pytest.raises(TypeError, match='real numbers'):
        runner.predict_action({}, '', policy_seed=0)


@pytest.mark.parametrize('width', [8, 14, 29])
@pytest.mark.parametrize(
    'dtype',
    [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64])
def test_serving_preserves_numeric_actions(width, dtype):
    values = torch.arange(3 * width).reshape(1, 3, width).to(dtype)
    runner = FluxVLAServingRunner(
        _Policy(values),
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True)
    actions, _ = runner.predict_action({}, '', policy_seed=0)
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, values[0].float().numpy())


def test_serving_keeps_denormalizer_intermediate_precision():
    values = np.array([[1.0 + 2**-30, 0.25]], dtype=np.float64)
    runner = FluxVLAServingRunner(
        _Policy(),
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=False,
        denormalize_action=lambda payload: values)
    # The helper must preserve the denormalizer's FP64 until canonicalization.
    intermediate = runner._call_denormalizer(np.zeros((1, 2)), {})
    assert intermediate.dtype == np.float64
    np.testing.assert_array_equal(intermediate, values)
    np.testing.assert_array_equal(
        runner._canonicalize_actions(intermediate), values.astype(np.float32))


def test_serving_runner_can_keep_parameters_fp32():
    model = _Policy()

    FluxVLAServingRunner(
        model,
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=True,
        keep_params_fp32=True,
        model_outputs_environment_actions=True,
    )

    assert model.dtype is None
    assert model.device == torch.device('cpu')
