# Copyright 2026 Limx Dynamics
"""Development checks for the configurable RoboLab observation boundary."""

import copy
import json

import numpy as np
import pytest
import torch

from fluxvla.datasets.robolab_dataset import RoboLabEvalDataset
from fluxvla.engines import build_dataset_from_cfg
from fluxvla.engines.runners.serving.serving_runner import FluxVLAServingRunner


class _Echo:

    def __call__(self, data):
        return data


class _RawPack:

    def __call__(self, data):
        return dict(
            frames=[data['shoulder'], data['wrist']],
            proprio=data['states'],
            text=data['task_description'])


class _NamedPack:

    def __call__(self, data):
        return dict(
            camera_map={name: data[name]
                        for name in ('shoulder', 'wrist')},
            robot_state=data['states'],
            instruction=data['task_description'])


class _Mutate:

    def __call__(self, data):
        data['shoulder'][:] = 0
        data['states'][:] = 7
        data['context']['episode_id'] = 'changed'
        return data


class _Affine:

    def __init__(self, scale=1, offset=0):
        self.scale = scale
        self.offset = offset

    def __call__(self, data):
        data['states'] = data['states'] * self.scale + self.offset
        return data


class _BadOutput:

    def __call__(self, data):
        return None


class _SometimesRaises:

    def __call__(self, data):
        if data['context'].get('fail'):
            raise RuntimeError('test transform failure')
        return data


class _TensorPack:

    def __call__(self, data):
        data['states'][:] = 0
        return dict(observation_vector=torch.from_numpy(data['states']))


class _Policy:

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def to(self, device, dtype=None):
        return self

    def predict_action(self, **batch):
        self.calls.append(batch)
        return np.zeros((1, 8), dtype=np.float32)


class _Decoder:

    def __init__(self):
        self.context = None

    def __call__(self, data):
        self.context = data
        return data['action']


@pytest.fixture
def observation():
    return dict(
        shoulder=np.full((8, 12, 3), 17, dtype=np.uint8),
        wrist=np.full((8, 12, 3), 93, dtype=np.uint8),
        states=np.array([.1, -.2, .3, -.4, .5, -.6, .7, .25],
                        dtype=np.float64),
        task_description='Place_the bowl\ncarefully',
        is_new_episode=True,
        context=dict(episode_id='first'))


def _dataset(**kwargs):
    cfg = dict(img_keys=['shoulder', 'wrist'], transforms=[dict(type=_Echo)])
    cfg.update(kwargs)
    return RoboLabEvalDataset(**cfg)


@pytest.mark.parametrize(('packer', 'keys'), [
    (_RawPack, {'frames', 'proprio', 'text'}),
    (_NamedPack, {'camera_map', 'robot_state', 'instruction'}),
])
def test_pipeline_selects_batch_fields_without_tokens(observation, packer,
                                                      keys):
    dataset = _dataset(transforms=[dict(type=packer)])
    batch, replay = dataset(observation)
    assert set(batch) == keys
    assert replay is None
    np.testing.assert_array_equal(dataset.last_raw_state,
                                  observation['states'])


@pytest.mark.parametrize('camera_count', [1, 2, 3])
@pytest.mark.parametrize('state_dim', [8, 14])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_preserves_configured_cameras_and_native_state(camera_count, state_dim,
                                                       dtype):
    keys = [f'camera_{index}' for index in range(camera_count)]
    observation = {
        name: np.full((7 + index, 13, 3), 20 + index, dtype=np.uint8)
        for index, name in enumerate(keys)
    }
    observation.update(
        states=np.arange(state_dim, dtype=dtype) / 10,
        task_description='Do_not change\nthis text')
    dataset = _dataset(img_keys=keys)
    batch, _ = dataset(observation)
    assert set(batch) == set(observation)
    for name in keys + ['states']:
        np.testing.assert_array_equal(batch[name], observation[name])
        assert batch[name].dtype == observation[name].dtype
        assert not np.shares_memory(batch[name], observation[name])
    assert batch['task_description'] == observation['task_description']
    assert dataset.last_raw_state.dtype == dtype
    assert not np.shares_memory(dataset.last_raw_state, batch['states'])


def test_inplace_transforms_and_returned_batch_do_not_modify_source(
        observation):
    before = copy.deepcopy(observation)
    dataset = _dataset(transforms=[dict(type=_Mutate)])
    batch, _ = dataset(observation)
    assert np.all(batch['states'] == 7)
    assert np.all(batch['shoulder'] == 0)
    batch['states'][:] = -99
    batch['wrist'][:] = 255
    np.testing.assert_array_equal(dataset.last_raw_state, before['states'])
    for key in ('states', 'shoulder', 'wrist'):
        np.testing.assert_array_equal(observation[key], before[key])
    assert observation['context'] == before['context']
    second, _ = dataset(observation)
    assert np.all(second['states'] == 7)
    np.testing.assert_array_equal(second['wrist'], before['wrist'])


def test_configured_transform_order(observation):
    dataset = _dataset(transforms=[
        dict(type=_Affine, offset=2),
        dict(type=_Affine, scale=3),
    ])
    batch, _ = dataset(observation)
    np.testing.assert_array_equal(batch['states'],
                                  (observation['states'] + 2) * 3)
    np.testing.assert_array_equal(dataset.last_raw_state,
                                  observation['states'])


@pytest.mark.parametrize('source', ['mapping', 'str', 'path'])
def test_statistics_selection_and_copies(observation, tmp_path, source):
    stats = {'left': {'values': [1, 2]}, 'right': {'values': [9, 8]}}
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(stats))
    configured = stats if source == 'mapping' else str(path)
    if source == 'path':
        configured = path
    dataset = _dataset(norm_stats=configured, unnorm_key='right')
    observation['norm_stats'] = {'values': [-1, -1]}
    batch, _ = dataset(observation)
    assert batch['norm_stats'] == {'values': [9, 8]}
    batch['norm_stats']['values'][0] = 99
    again, _ = dataset(observation)
    assert again['norm_stats'] == {'values': [9, 8]}
    assert stats['right']['values'] == [9, 8]
    assert observation['norm_stats'] == {'values': [-1, -1]}


def test_request_cannot_supply_unconfigured_statistics(observation):
    observation['norm_stats'] = {'values': [999]}
    batch, _ = _dataset()(observation)
    assert 'norm_stats' not in batch
    assert observation['norm_stats'] == {'values': [999]}


@pytest.mark.parametrize(('key', 'error'), [
    (None, ValueError),
    ('missing', KeyError),
])
def test_statistics_require_explicit_existing_key(key, error):
    with pytest.raises(error, match='unnorm_key|missing'):
        _dataset(norm_stats={'chosen': {'values': [1]}}, unnorm_key=key)


@pytest.mark.parametrize('missing', ['shoulder', 'states', 'task_description'])
def test_missing_field_clears_previous_raw_state(observation, missing):
    dataset = _dataset()
    dataset(observation)
    del observation[missing]
    with pytest.raises(KeyError, match=missing):
        dataset(observation)
    assert dataset.last_raw_state is None


@pytest.mark.parametrize(('key', 'value', 'error'), [
    ('shoulder', np.zeros((8, 12, 4), dtype=np.uint8), ValueError),
    ('shoulder', np.zeros((0, 12, 3), dtype=np.uint8), ValueError),
    ('shoulder', np.zeros((8, 12, 3), dtype=np.float32), TypeError),
    ('states', np.zeros((1, 8)), ValueError),
    ('states', np.array([]), ValueError),
    ('states', np.array([np.nan]), ValueError),
    ('states', np.array([np.inf]), ValueError),
    ('states', np.array([1 + 2j]), ValueError),
    ('states', np.array(['0.1']), ValueError),
    ('task_description', None, TypeError),
])
def test_invalid_observation_has_field_error_and_no_stale_state(
        observation, key, value, error):
    dataset = _dataset()
    dataset(observation)
    observation[key] = value
    with pytest.raises(error, match=key):
        dataset(observation)
    assert dataset.last_raw_state is None


def test_non_mapping_and_failed_transform_clear_state(observation):
    dataset = _dataset()
    dataset(observation)
    with pytest.raises(TypeError, match='mapping'):
        dataset(None)
    assert dataset.last_raw_state is None
    bad_output = _dataset(transforms=[dict(type=_BadOutput)])
    with pytest.raises(TypeError, match='transform|Transform'):
        bad_output(observation)
    assert bad_output.last_raw_state is None
    raising = _dataset(transforms=[dict(type=_SometimesRaises)])
    raising(observation)
    observation['context']['fail'] = True
    with pytest.raises(RuntimeError, match='test transform failure'):
        raising(observation)
    assert raising.last_raw_state is None


@pytest.mark.parametrize(('kwargs', 'error', 'message'), [
    ({
        'img_keys': []
    }, ValueError, 'img_keys'),
    ({
        'img_keys': 'shoulder'
    }, TypeError, 'img_keys'),
    ({
        'img_keys': ['shoulder', 'shoulder']
    }, ValueError, 'img_keys'),
    ({
        'img_keys': ['shoulder', 3]
    }, TypeError, 'img_keys'),
    ({
        'state_key': ''
    }, ValueError, 'state_key'),
    ({
        'transforms': []
    }, ValueError, 'transforms'),
    ({
        'transforms': [None]
    }, TypeError, 'transforms'),
])
def test_invalid_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        _dataset(**kwargs)


def test_configurable_field_names_and_context(observation):
    observation['robot_vector'] = observation.pop('states').tolist()
    observation['instruction'] = observation.pop('task_description')
    dataset = _dataset(state_key='robot_vector', task_key='instruction')
    first, _ = dataset(observation)
    assert first['is_new_episode'] is True
    assert first['context']['episode_id'] == 'first'
    assert 'reset_history' not in first
    observation['is_new_episode'] = False
    observation['context']['episode_id'] = 'second'
    second, _ = dataset(observation)
    assert second['is_new_episode'] is False
    assert second['context']['episode_id'] == 'second'
    assert first['context']['episode_id'] == 'first'
    assert second['instruction'] == observation['instruction']
    np.testing.assert_array_equal(dataset.last_raw_state,
                                  observation['robot_vector'])


def test_registry_and_existing_service_keep_raw_state_on_cpu(
        observation, monkeypatch):

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError('Observation pipeline must not initialize CUDA')

    monkeypatch.setattr(torch.Tensor, 'cuda', forbidden_cuda)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden_cuda)
    dataset = build_dataset_from_cfg(
        dict(
            type='RoboLabEvalDataset',
            img_keys=['shoulder', 'wrist'],
            transforms=[dict(type=_TensorPack)]))
    policy = _Policy()
    decoder = _Decoder()
    runner = FluxVLAServingRunner(
        vla=policy,
        dataset=dataset,
        denormalize_action=decoder,
        device='cpu',
        enable_mixed_precision=False,
        keep_params_fp32=True)
    actions, _ = runner.predict_action(observation, '', policy_seed=17)
    assert actions.shape == (1, 8)
    tensor = policy.calls[0]['observation_vector']
    assert tensor.device.type == 'cpu'
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor) == 0
    np.testing.assert_array_equal(decoder.context['state'],
                                  observation['states'].astype(np.float32))
    np.testing.assert_array_equal(dataset.last_raw_state,
                                  observation['states'])
    invalid = dict(observation, states=np.array([np.nan]))
    with pytest.raises(ValueError, match='states'):
        runner.predict_action(invalid, '', policy_seed=17)
    assert len(policy.calls) == 1
    assert dataset.last_raw_state is None
    assert not torch.cuda.is_initialized()
