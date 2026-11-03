# Copyright 2026 Limx Dynamics
"""Action compatibility options and CPU serving, without model assets."""

import copy
import json

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.datasets.robolab_dataset import RoboLabEvalDataset
from fluxvla.engines import build_transform_from_cfg
from fluxvla.engines.runners.serving.serving_runner import FluxVLAServingRunner
from fluxvla.transforms.normalize import (DenormalizeDeltaAction,
                                          DenormalizeLiberoAction,
                                          DenormalizePrivateAction)
from fluxvla.transforms.transform_inputs import PackModelInputs  # noqa: F401

DECODERS = [
    DenormalizeLiberoAction, DenormalizePrivateAction, DenormalizeDeltaAction
]


def stats(width):
    return dict(
        mean=[0.5] * width,
        std=[2.] * width,
        min=[-2.] * width,
        max=[2.] * width,
        q01=[-2.] * width,
        q99=[2.] * width)


def make_decoder(cls=DenormalizeDeltaAction, width=3, fields=None, **options):
    config = dict(
        norm_stats={'robot': {
            'actions': fields or stats(width)
        }},
        action_dim=width,
        norm_type='quantile',
        normalize_gripper_action=False,
        invert_gripper_action=False)
    if issubclass(cls, DenormalizePrivateAction):
        config['statistic_name'] = 'robot'
    if issubclass(cls, DenormalizeDeltaAction):
        config['delta_action_mask'] = [True] * (width - 1) + [False]
    config.update(options)
    return cls(**config)


def payload(action, state=None):
    return dict(
        action=np.asarray(action),
        norm_stats_key='robot',
        state=np.zeros(3) if state is None else state)


@pytest.mark.parametrize('cls', DECODERS)
@pytest.mark.parametrize('epsilon', [0., .125])
def test_quantile_endpoints_use_configured_epsilon(cls, epsilon):
    # Exact binary fractions isolate endpoint semantics from cancellation.
    decoder = make_decoder(cls, quantile_epsilon=epsilon)
    actual = decoder(payload(np.array([-1., 0., 1., 999.], np.float32)))
    expected = np.array([-2., epsilon / 2, 2. + epsilon])
    if cls is DenormalizeDeltaAction:
        expected = expected.astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('cls', DECODERS)
@pytest.mark.parametrize('clip', [False, True])
def test_quantile_preserves_mask_and_explicit_clipping(cls, clip):
    decoder = make_decoder(
        cls,
        quantile_epsilon=1e-6,
        action_norm_mask=[True, False, True],
        clip_normalized_action=clip)
    actual = decoder(payload(np.array([-2., .25, 2.], np.float64)))
    expected = np.array([-2., .25, 2. +
                         1e-6] if clip else [-4. - .5e-6, .25, 4. + 1.5e-6])
    if cls is DenormalizeDeltaAction:
        expected = expected.astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('cls', DECODERS)
@pytest.mark.parametrize('norm_type', ['mean_std', 'min_max'])
def test_epsilon_does_not_change_other_normalization_modes(cls, norm_type):
    data = payload(np.array([-.5, .25, 1.25], np.float32))
    old = make_decoder(cls, norm_type=norm_type)(copy.deepcopy(data))
    actual = make_decoder(
        cls, norm_type=norm_type, quantile_epsilon=.125)(
            copy.deepcopy(data))
    np.testing.assert_array_equal(actual, old)
    assert actual.dtype == old.dtype


@pytest.mark.parametrize('cls', DECODERS)
@pytest.mark.parametrize('value', [None, True, '1e-6', [1e-6], 1j])
def test_epsilon_rejects_non_real_scalar_values(cls, value):
    with pytest.raises(TypeError, match='quantile_epsilon'):
        make_decoder(cls, quantile_epsilon=value)


@pytest.mark.parametrize('cls', DECODERS)
@pytest.mark.parametrize('value', [-1e-6, np.inf, -np.inf, np.nan])
def test_epsilon_rejects_negative_or_nonfinite_values(cls, value):
    with pytest.raises(ValueError, match='quantile_epsilon'):
        make_decoder(cls, quantile_epsilon=value)


@pytest.mark.parametrize('cls', DECODERS)
def test_numpy_scalar_epsilon_and_json_stats(cls, tmp_path):
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps({'robot': {'action': stats(3)}}))
    decoder = make_decoder(
        cls, norm_stats=str(path), quantile_epsilon=np.float64(1e-6))
    actual = decoder(payload(np.zeros(3, np.float32)))
    np.testing.assert_allclose(actual, np.full(3, .5e-6), rtol=1e-7, atol=0)


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_precision_option_requires_boolean(value):
    with pytest.raises(TypeError, match='preserve_intermediate_dtype'):
        make_decoder(preserve_intermediate_dtype=value)


def test_rounding_before_state_addition_changes_final_fp32():
    # A fixed observed rounding boundary; no external asset is needed.
    fields = dict(
        q01=[-0.28571800187826146, 0.], q99=[0.2831991063714028, .9998])
    data = payload(
        np.array([.042880743741989136, 0.], np.float32),
        np.array([.012485947459936142, .8], np.float32))
    early = make_decoder(
        width=2, fields=fields, quantile_epsilon=1e-6)(
            copy.deepcopy(data))
    late = make_decoder(
        width=2,
        fields=fields,
        quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True)(
            copy.deepcopy(data))
    assert early.dtype == np.float32
    assert late.dtype == np.float64
    assert early[0] == np.float32(.023424822837114334)
    assert np.float32(late[0]) == np.float32(.023424824699759483)
    assert early[0] != np.float32(late[0])
    assert early[1] == np.float32(late[1])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_precision_option_retains_existing_intermediate_dtype(dtype):
    fields = {k: np.asarray(v, dtype=dtype) for k, v in stats(3).items()}
    data = payload(
        np.array([0., .25, .5], dtype), np.array([1., 2., 3.], dtype))
    result = make_decoder(
        fields=fields, preserve_intermediate_dtype=True)(
            data)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, [1., 2.5, 1.])


@pytest.mark.parametrize('epsilon', [0., 1e-6])
@pytest.mark.parametrize('preserve', [False, True])
def test_constant_statistics_keep_epsilon_and_absolute_dimensions(
        epsilon, preserve):
    decoder = make_decoder(
        fields=dict(q01=[0.] * 3, q99=[0.] * 3),
        quantile_epsilon=epsilon,
        preserve_intermediate_dtype=preserve)
    actual = decoder(
        payload(
            np.array([-2., 0., 2.], np.float32),
            np.array([1., 2., 999.], np.float32)))
    expected = np.array([1. - .5 * epsilon, 2. + .5 * epsilon, 1.5 * epsilon])
    np.testing.assert_array_equal(
        actual, expected if preserve else expected.astype(np.float32))


@pytest.mark.parametrize('width,horizon', [(2, 1), (8, 15), (14, 3), (29, 31)])
def test_generic_width_horizon_mask_permutation_and_padding(width, horizon):
    rng = np.random.default_rng(7)
    physical = rng.uniform(-.2, .2, (horizon, width))
    physical[0] = 0.
    state = rng.uniform(-1, 1, width)
    mask = np.arange(width) % 3 != 2
    permutation = list(reversed(range(width)))
    normalized = (physical + 2.) / (4. + 1e-6) * 2. - 1.
    padded = np.pad(normalized, ((0, 0), (0, 5)), constant_values=1e9)
    decoder = make_decoder(
        width=width,
        quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True,
        delta_action_mask=mask.tolist(),
        state_permutation=permutation)
    before = copy.deepcopy((padded, state, decoder.norm_stats))
    actual = np.asarray(decoder(payload(padded,
                                        state))).reshape(horizon, width)
    expected = physical + np.where(mask, state[permutation], 0.)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(padded, before[0])
    np.testing.assert_array_equal(state, before[1])
    assert decoder.norm_stats == before[2]
    assert not np.shares_memory(actual, padded)


@pytest.mark.parametrize('shape', [(3, ), (1, 3), (1, 1, 3)])
def test_single_action_shapes_and_state_batch(shape):
    decoder = make_decoder(
        quantile_epsilon=.125, preserve_intermediate_dtype=True)
    actual = decoder(payload(np.zeros(shape), np.array([[1., 2., 100.]])))
    np.testing.assert_array_equal(
        np.asarray(actual).reshape(-1), [1.0625, 2.0625, .0625])


def test_new_precision_keyword_does_not_shift_legacy_positional_arguments():
    decoder = DenormalizeDeltaAction([True, True, False],
                                     None, {'robot': {
                                         'actions': stats(3)
                                     }},
                                     3,
                                     'quantile',
                                     statistic_name='robot',
                                     preserve_intermediate_dtype=True)
    np.testing.assert_array_equal(decoder(payload(np.zeros(3))), np.zeros(3))


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize(
    'case', ['missing_state', 'state_rank', 'state_width', 'action_batch'])
def test_existing_shape_errors_are_preserved(preserve, case):
    decoder = make_decoder(preserve_intermediate_dtype=preserve)
    data = payload(np.zeros(3))
    match = 'state'
    if case == 'missing_state':
        data.pop('state')
    elif case == 'state_rank':
        data['state'] = np.zeros((2, 3))
    elif case == 'state_width':
        data['state'] = np.zeros(2)
    else:
        data['action'] = np.zeros((2, 1, 3))
        match = 'batch'
    with pytest.raises(ValueError, match=match):
        decoder(data)


def test_horizon_statistics_and_discrete_mode_keep_their_semantics():
    fields = dict(
        q01=[[-2., -2.], [-4., -4.]],
        q99=[[2., 2.], [6., 6.]],
        min=[[0., 0.], [0., 0.]],
        max=[[1., 1.], [2., 2.]])
    decoder = make_decoder(
        width=2,
        fields=fields,
        quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True,
        discrete_action_dims=[1],
        discrete_norm_type='min_max')
    data = payload(np.zeros(2), np.array([2., 100.]))
    data['action_horizon_index'] = 1
    assert decoder.get_action_stats_horizon() == 2
    np.testing.assert_allclose(
        decoder(data), [3. + .5e-6, 1.], rtol=0, atol=1e-14)
    assert decoder.action_norm_mask is None


def test_gripper_threshold_difference_survives_final_fp32():
    fields = dict(q01=[-2., -2., 0.], q99=[2., 2., .9998])
    data = payload(np.array([0., 0., .0001991583703784272], np.float32))
    old = make_decoder(fields=fields)(copy.deepcopy(data))
    new = make_decoder(
        fields=fields, quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True)(
            copy.deepcopy(data))
    assert old[2] <= .5
    assert np.float32(new[2]) > .5
    assert np.float32(new[2]) == np.float32(.5000000596046448)


class ReplayModel(torch.nn.Module):

    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def predict_action(self, **batch):
        assert batch['states'].device.type == 'cpu'
        assert batch['reset_history'] is True
        return torch.from_numpy(self.actions.copy())


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('per_action', [False, True])
def test_registered_decoder_in_actual_robolab_serving_runner(
        horizon, per_action):
    # Model-specific text/images are covered by the input pipeline tests.
    dataset = RoboLabEvalDataset(
        img_keys=['camera'],
        transforms=[
            dict(
                type='PackModelInputs',
                key_mapping={'states': 'states'},
                dtypes={'states': 'float32'},
                add_batch_dim=True)
        ])
    options = dict(
        type='DenormalizeDeltaAction',
        norm_stats={'robot': {
            'actions': stats(3)
        }},
        statistic_name='robot',
        action_dim=3,
        norm_type='quantile',
        delta_action_mask=[True, True, False],
        quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True)
    cfg = Config.fromstring('decoder = ' + repr(options), '.py')
    decoder = build_transform_from_cfg(cfg.decoder)
    runner = FluxVLAServingRunner(
        ReplayModel(np.zeros((1, horizon, 7), np.float32)),
        dataset,
        decoder,
        device='cpu',
        enable_mixed_precision=False,
        keep_params_fp32=True,
        denormalize_per_action=per_action,
        expected_unnorm_key='robot')
    obs = dict(
        camera=np.zeros((2, 2, 3), np.uint8),
        states=np.array([1., 2., 99.], np.float32),
        task_description='Test')
    before = copy.deepcopy(obs)
    actual, _ = runner.predict_action(obs, unnorm_key='robot', policy_seed=7)
    expected = np.tile(
        np.array([1. + .5e-6, 2. + .5e-6, .5e-6], np.float32), (horizon, 1))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(obs['states'], before['states'])
    np.testing.assert_array_equal(obs['camera'], before['camera'])
    np.testing.assert_array_equal(dataset.last_raw_state, before['states'])
    assert not np.shares_memory(dataset.last_raw_state, obs['states'])
