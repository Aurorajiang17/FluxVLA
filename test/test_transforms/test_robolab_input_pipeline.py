# Copyright 2026 Limx Dynamics
"""Opt-in preprocessing and configurable CPU packing, without model assets."""

import copy

import numpy as np
import pytest
import torch
from mmengine import Config
from PIL import Image

from fluxvla.datasets.robolab_dataset import RoboLabEvalDataset
from fluxvla.engines import build_dataset_from_cfg, build_transform_from_cfg
from fluxvla.tokenizers import paligemma_tokenizer as tokenizer_module
from fluxvla.transforms.normalize import PadStatesAndActions, StateFromInputs
from fluxvla.transforms.prompters import PreparePromptWithState
from fluxvla.transforms.transform_images import TransformImage
from fluxvla.transforms.transform_inputs import ProcessEvalInputs
from fluxvla.transforms.transform_prompts import ProcessPrompts


def state_stats(width):
    return dict(
        mean=[0.0] * width,
        std=[1.0] * width,
        min=[-1.0] * width,
        max=[1.0] * width,
        q01=[-1.0] * width,
        q99=[1.0] * width)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('norm_type', ['quantile', 'mean_std', 'min_max'])
def test_state_precision_and_optional_padding(dtype, preserve, norm_type):
    raw = np.array([1.0 + 1e-10, -2.0, 0.25], dtype=dtype)
    original = raw.copy()
    stats = state_stats(3)
    normalized_input = raw if preserve else raw.astype(np.float32)
    if norm_type == 'quantile':
        expected = (normalized_input - np.asarray(stats['q01'])) / (
            np.asarray(stats['q99']) - np.asarray(stats['q01']) + 1e-6) * 2 - 1
    elif norm_type == 'mean_std':
        expected = (normalized_input - np.asarray(stats['mean'])) / (
            np.asarray(stats['std']) + 1e-6)
    else:
        expected = (normalized_input - np.asarray(stats['min'], np.float32)
                    ) / (np.asarray(stats['max'], np.float32) -
                         np.asarray(stats['min'], np.float32) + 1e-6) * 2 - 1
    unpadded = StateFromInputs(
        norm_type=norm_type, preserve_input_dtype=preserve)(dict(
            states=raw, norm_stats={'proprio': stats}))['states']
    np.testing.assert_array_equal(unpadded, expected)
    padded = StateFromInputs(
        norm_type=norm_type, preserve_input_dtype=preserve,
        state_dim=5)(dict(states=raw, norm_stats={'proprio': stats}))['states']
    expected_dtype = expected.dtype if preserve else np.dtype('float32')
    assert padded.dtype == expected_dtype
    np.testing.assert_array_equal(padded[:3], expected.astype(expected_dtype))
    np.testing.assert_array_equal(padded[3:], [0, 0])
    np.testing.assert_array_equal(raw, original)


def test_state_default_keeps_fp32_input_and_padding_behavior():
    raw = np.array([1.0 + 1e-10, -0.123456789], dtype=np.float64)
    data = dict(states=raw, norm_stats={'proprio': state_stats(2)})
    old_default = StateFromInputs(norm_type='quantile', state_dim=4)(data)
    explicit = StateFromInputs(
        norm_type='quantile', state_dim=4, preserve_input_dtype=False)(
            data)
    np.testing.assert_array_equal(old_default['states'], explicit['states'])
    assert old_default['states'].dtype == np.float32


def test_quantile_keeps_epsilon_constant_dimensions_and_unclipped_values():
    data = dict(
        states=np.array([2.0, -2.0], dtype=np.float64),
        norm_stats={'state': dict(q01=[0.0, 0.0], q99=[0.0, 0.0])})
    result = StateFromInputs(
        stat_key='state', norm_type='quantile', preserve_input_dtype=True)(
            data)
    np.testing.assert_array_equal(result['states'], [3999999, -4000001])


def test_state_stats_must_match_native_width():
    with pytest.raises(ValueError, match='width'):
        StateFromInputs(
            norm_type='quantile', preserve_input_dtype=True)(
                dict(
                    states=np.zeros(8),
                    norm_stats={'proprio': state_stats(32)}))


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_state_precision_option_rejects_non_boolean(value):
    with pytest.raises(TypeError, match='preserve_input_dtype'):
        StateFromInputs(preserve_input_dtype=value)


def test_state_prompt_precedes_padding():
    data = dict(states=np.zeros(2), task_description='Pick_up cup')
    data = PreparePromptWithState()(data)
    assert data['prompt'] == 'Task: Pick up cup, State: 128 128;\nAction: '
    data = PadStatesAndActions(32)(data)
    assert data['states'].shape == (32, )
    assert data['prompt'].count('128') == 2


def image_transform(**kwargs):
    return TransformImage(
        input_sizes=[[3, 224, 224]],
        means=[[127.5] * 3],
        stds=[[127.5] * 3],
        letterbox_fill=[0, 0, 0],
        letterbox_pad_position='center',
        **kwargs)


@pytest.mark.parametrize('shape, round_size, floor_size', [
    ((301, 500), (135, 224), (134, 224)),
    ((500, 301), (224, 135), (224, 134)),
    ((480, 640), (168, 224), (168, 224)),
])
def test_letterbox_rounding_and_center_padding(shape, round_size, floor_size):
    image = Image.fromarray(np.full((*shape, 3), 255, dtype=np.uint8))
    old_default = image_transform().letterbox_pad_transform(image, [0, 0, 0])
    explicit = image_transform(
        letterbox_resize_rounding='round').letterbox_pad_transform(
            image, [0, 0, 0])
    np.testing.assert_array_equal(old_default, explicit)
    for mode, size in [('round', round_size), ('floor', floor_size)]:
        result = image_transform(letterbox_resize_rounding=mode)(
            dict(pixel_values=[image]))['pixel_values']
        expected = torch.full((3, 224, 224), -1.0)
        h, w = size
        top, left = (224 - h) // 2, (224 - w) // 2
        expected[:, top:top + h, left:left + w] = 1.0
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_floor_rejects_zero_sized_resize_without_changing_round_mode():
    image = Image.fromarray(np.zeros((1, 1000, 3), dtype=np.uint8))
    assert image_transform().letterbox_pad_transform(image,
                                                     [0, 0, 0]).size == (224,
                                                                         224)
    with pytest.raises(ValueError, match='zero'):
        image_transform(letterbox_resize_rounding='floor')(
            dict(pixel_values=[image]))


@pytest.mark.parametrize('value', ['ceil', None, 1])
def test_letterbox_rejects_unknown_rounding(value):
    with pytest.raises(ValueError, match='letterbox_resize_rounding'):
        image_transform(letterbox_resize_rounding=value)


@pytest.mark.parametrize('padding', [0, 1, 2])
def test_eval_images_preserve_real_cameras_and_mask_padding(padding):
    shoulder = np.full((4, 5, 3), 25, np.uint8)
    wrist = np.full((6, 7, 3), 200, np.uint8)
    result = ProcessEvalInputs(
        img_keys=['shoulder', 'wrist'], num_padding_imgs=padding)(
            dict(shoulder=shoulder, wrist=wrist))
    assert result['img_masks'] == [True, True] + [False] * padding
    np.testing.assert_array_equal(result['pixel_values'][0], shoulder)
    np.testing.assert_array_equal(result['pixel_values'][1], wrist)
    for image in result['pixel_values'][2:]:
        np.testing.assert_array_equal(image, np.zeros_like(shoulder))


def test_eval_default_does_not_add_images_or_hide_missing_cameras():
    obs = dict(camera=np.zeros((4, 5, 3), np.uint8))
    result = ProcessEvalInputs(img_keys=['camera'])(copy.deepcopy(obs))
    explicit = ProcessEvalInputs(
        img_keys=['camera'], num_padding_imgs=0)(
            copy.deepcopy(obs))
    assert result['img_masks'] == explicit['img_masks'] == [True]
    np.testing.assert_array_equal(result['pixel_values'][0],
                                  explicit['pixel_values'][0])
    with pytest.raises(KeyError, match='missing'):
        ProcessEvalInputs(
            img_keys=['camera', 'missing'], num_padding_imgs=1)(
                obs)


@pytest.mark.parametrize('value', [-1, True, 1.5, '1'])
def test_eval_padding_requires_a_nonnegative_integer(value):
    with pytest.raises((TypeError, ValueError), match='num_padding_imgs'):
        ProcessEvalInputs(img_keys=['camera'], num_padding_imgs=value)


def test_padding_requires_a_real_image_as_the_shape_reference():
    with pytest.raises(ValueError, match='image'):
        ProcessEvalInputs(img_keys=[], num_padding_imgs=1)({})


def packer(**kwargs):
    return build_transform_from_cfg(dict(type='PackModelInputs', **kwargs))


@pytest.mark.parametrize('add_batch_dim', [False, True])
def test_pack_uses_configured_fields_and_dtypes_without_language_inputs(
        add_batch_dim):
    transform = packer(
        key_mapping={
            'proprio': 'raw_state',
            'valid': 'validity'
        },
        dtypes={
            'proprio': 'float32',
            'valid': 'bool'
        },
        add_batch_dim=add_batch_dim)
    result = transform(
        dict(
            raw_state=np.arange(5, dtype=np.float64),
            validity=[1, 0],
            unused='not a model input'))
    assert set(result) == {'proprio', 'valid'}
    assert result['proprio'].shape == ((1, 5) if add_batch_dim else (5, ))
    assert result['valid'].shape == ((1, 2) if add_batch_dim else (2, ))
    assert result['proprio'].dtype == torch.float32
    assert result['valid'].dtype == torch.bool
    assert all(t.device.type == 'cpu' for t in result.values())


@pytest.mark.parametrize('value', ['text', ['bad'], {'nested': 1}])
def test_pack_rejects_nonnumeric_input(value):
    transform = packer(key_mapping={'out': 'in'}, dtypes={'out': 'float32'})
    with pytest.raises(TypeError, match='numeric'):
        transform({'in': value})


def test_pack_rejects_missing_fields_and_non_cpu_tensors():
    transform = packer(key_mapping={'out': 'in'}, dtypes={'out': 'float32'})
    with pytest.raises(KeyError, match='in'):
        transform({})
    with pytest.raises(ValueError, match='CPU'):
        transform({'in': torch.empty(2, device='meta')})


@pytest.mark.parametrize('kwargs', [
    dict(key_mapping={}, dtypes={}),
    dict(key_mapping={'out': ''}, dtypes={'out': 'float32'}),
    dict(key_mapping={'out': 'in'}, dtypes={}),
    dict(key_mapping={'out': 'in'}, dtypes={'wrong': 'float32'}),
    dict(key_mapping={'out': 'in'}, dtypes={'out': 'not_a_dtype'}),
    dict(
        key_mapping={'out': 'in'}, dtypes={'out': 'float32'}, add_batch_dim=1),
])
def test_pack_rejects_ambiguous_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        packer(**kwargs)


def test_configuration_builds_the_complete_cpu_input_chain(
        tmp_path, monkeypatch):
    asset = tmp_path / 'tokenizer.model'
    asset.write_bytes(b'local-test-processor')
    prompts = []

    class Recorder:

        def encode(self, text, **kwargs):
            prompts.append(text)
            return [2, 41, 42]

    monkeypatch.setattr(tokenizer_module, 'maybe_download',
                        lambda *a, **k: asset)
    monkeypatch.setattr(tokenizer_module.sentencepiece,
                        'SentencePieceProcessor', lambda **kwargs: Recorder())
    # Importing these production classes registers the same string names used
    # by normal configs; the CPU test runtime skips only package initializers.
    cfg = Config.fromstring(
        "eval = dict(dataset=dict(type='RoboLabEvalDataset', "
        "img_keys=['shoulder', 'wrist'], "
        "unnorm_key='test_robot', transforms=[]))", '.py')
    cfg.eval.dataset.norm_stats = {'test_robot': {'state': state_stats(8)}}
    cfg.eval.dataset.transforms = [
        dict(
            type='ProcessEvalInputs',
            img_keys=['shoulder', 'wrist'],
            num_padding_imgs=1),
        dict(
            type='StateFromInputs',
            stat_key='state',
            norm_type='quantile',
            preserve_input_dtype=True),
        dict(type='PreparePromptWithState'),
        dict(
            type='ProcessPrompts',
            max_len=200,
            tokenizer=dict(
                type='PaligemmaTokenizer',
                model_path=str(asset),
                preformatted=True)),
        dict(type='PadStatesAndActions', model_action_dim=32),
        dict(
            type='TransformImage',
            input_sizes=[[3, 224, 224]] * 3,
            means=[[127.5] * 3] * 3,
            stds=[[127.5] * 3] * 3,
            letterbox_fill=[0, 0, 0],
            letterbox_resize_rounding='floor'),
        dict(
            type='PackModelInputs',
            key_mapping={
                'images': 'pixel_values',
                'states': 'states',
                'img_masks': 'img_masks',
                'lang_tokens': 'lang_tokens',
                'lang_masks': 'lang_masks'
            },
            dtypes={
                'images': 'float32',
                'states': 'float32',
                'img_masks': 'bool',
                'lang_tokens': 'int64',
                'lang_masks': 'bool'
            }),
    ]
    dataset = build_dataset_from_cfg(cfg.eval.dataset)
    assert isinstance(dataset, RoboLabEvalDataset)
    assert any(isinstance(t, ProcessPrompts) for t in dataset.transforms)
    obs = dict(
        shoulder=np.full((301, 500, 3), 255, np.uint8),
        wrist=np.zeros((301, 500, 3), np.uint8),
        states=np.zeros(8),
        task_description='Pick_up cup')
    before = copy.deepcopy(obs)
    batch, replay = dataset(obs)
    assert replay is None
    assert prompts == [
        'Task: Pick up cup, State: ' + ' '.join(['127'] * 8) + ';\nAction: '
    ]
    assert batch['images'].shape == (1, 9, 224, 224)
    assert batch['states'].shape == (1, 32)
    assert batch['lang_tokens'].shape == (1, 200)
    assert batch['img_masks'].tolist() == [[True, True, False]]
    assert batch['lang_tokens'][0, :3].tolist() == [2, 41, 42]
    assert batch['lang_masks'].sum().item() == 3
    assert torch.all(batch['images'][:, 6:] == -1)
    assert all(t.device.type == 'cpu' for t in batch.values())
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])
    np.testing.assert_array_equal(dataset.last_raw_state, before['states'])
    assert not np.shares_memory(dataset.last_raw_state, obs['states'])
    assert not torch.cuda.is_initialized()
