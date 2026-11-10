# Copyright 2026 Limx Dynamics
"""Recipe and train/serve input consistency of the DROID LeRobot config."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from fluxvla.engines import build_dataset_from_cfg, build_transform_from_cfg
from fluxvla.optimizers.lr_scheduler_policies import \
    LinearWarmupCosineDecayLRScheduler
from fluxvla.tokenizers import paligemma_tokenizer as tokenizer_module

# The CPU test runtime skips package initializers; importing the production
# modules registers the names used by both configs.
for _module in ('fluxvla.datasets.robolab_dataset',
                'fluxvla.transforms.normalize', 'fluxvla.transforms.prompters',
                'fluxvla.transforms.transform_actions',
                'fluxvla.transforms.transform_images',
                'fluxvla.transforms.transform_inputs',
                'fluxvla.transforms.transform_prompts'):
    importlib.import_module(_module)

CONFIGS = Path(__file__).resolve().parents[2] / 'configs/pi05'
TRAIN = CONFIGS / 'pi05_paligemma_droid_lerobot_full_finetune.py'
SERVE = CONFIGS / 'pi05_paligemma_robolab_droid_lerobot_inference.py'
RELEASED = [
    CONFIGS / 'pi05_paligemma_robolab_droid_jointpos_inference.py',
    CONFIGS / 'pi05_paligemma_robolab_droid_jointpos_process_inference.py',
]
SERVE_ENV = ('FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT',
             'FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER',
             'FLUXVLA_ROBOLAB_DROID_FT_STATS')
BASE_DROID_STATS_SHA256 = (
    '508e75bb8f28f86ad8998a2254d2076fdd42caf26d8a89baecd5de0f628e2280')


@pytest.fixture(scope='module')
def cfg():
    return Config.fromfile(str(TRAIN))


def test_recipe_matches_openpi_full_droid(cfg):
    runner = cfg.runner
    assert runner.type == 'FSDPTrainRunner'
    assert runner.strict_training_state is True
    assert runner.resume_data_progress is True
    assert runner.ema_decay == 0.99
    assert runner.max_steps == 100000 and runner.max_epochs is None
    assert runner.max_grad_norm == 1.0
    optimizer = runner.optimizer
    assert optimizer.type == 'AdamW' and optimizer.lr == 5e-5
    assert list(optimizer.betas) == [0.9, 0.95]
    assert optimizer.eps == 1e-8 and optimizer.weight_decay == 1e-10
    assert runner.keep_params_fp32 and runner.mixed_precision_dtype == 'bf16'
    # The runner enforces the global batch for any GPU count.
    assert runner.expected_global_batch_size == 256
    assert (runner.save_iter_interval, runner.max_keep_ckpts,
            runner.milestone_interval) == (5000, 2, 10000)
    model = cfg.model
    assert model.type == 'PI05FlowMatching'
    assert model.n_action_steps == 16 and model.loss_action_dim == 32
    assert model.max_action_dim == 32 and model.ori_action_dim == 8
    assert (model.time_sampler, model.time_beta_alpha,
            model.time_beta_beta) == ('beta', 1.5, 1.0)
    assert model.pretrained_name_or_path
    assert model.strict_mapping is True
    # Serving of the trained model uses the training horizon.
    assert cfg.inference_model.n_action_steps == 16
    assert cfg.inference_model.pretrained_name_or_path is None


def _serving_config(monkeypatch, **environment):
    for name in SERVE_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return Config.fromfile(str(SERVE))


def test_trained_model_serving_uses_16_steps_and_own_assets(monkeypatch):
    serve = _serving_config(monkeypatch)
    assert serve.model.n_action_steps == 16
    assert serve.inference_model.n_action_steps == 16
    # Without an explicit file the run's dataset_statistics.json is used.
    assert serve.themis.server.norm_stats_path is None
    assert serve.themis.server.runner.sampling_mode == (
        'process_noise_stream_v1')
    assert 'pi05_droid_jointpos' not in serve.themis.server.ckpt_path
    serve = _serving_config(
        monkeypatch,
        FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT='/run/model.safetensors',
        FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER='/run/tokenizer.model',
        FLUXVLA_ROBOLAB_DROID_FT_STATS='/run/stats.json')
    assert serve.themis.server.ckpt_path == '/run/model.safetensors'
    assert serve.themis.server.norm_stats_path == '/run/stats.json'
    assert serve.eval.dataset.norm_stats == '/run/stats.json'
    assert serve.eval.denormalize_action.norm_stats == '/run/stats.json'
    prompts = [
        spec for spec in serve.eval.dataset.transforms
        if spec['type'] == 'ProcessPrompts'
    ]
    assert prompts[0]['tokenizer']['model_path'] == '/run/tokenizer.model'


def test_released_model_configs_keep_15_steps(monkeypatch):
    serve = _serving_config(monkeypatch)
    for path in RELEASED:
        released = Config.fromfile(str(path))
        assert released.model.n_action_steps == 15
        assert released.inference_model.n_action_steps == 15
        # The new entry changes only the horizon and asset paths.
        new_model = copy.deepcopy(serve.inference_model.to_dict())
        old_model = copy.deepcopy(released.inference_model.to_dict())
        assert new_model.pop('n_action_steps') == 16
        assert old_model.pop('n_action_steps') == 15
        assert new_model == old_model
        new_transforms = copy.deepcopy(serve.eval.dataset.transforms)
        old_transforms = copy.deepcopy(released.eval.dataset.transforms)
        for transforms in (new_transforms, old_transforms):
            for spec in transforms:
                if spec['type'] == 'ProcessPrompts':
                    spec['tokenizer'].pop('model_path')
        assert new_transforms == old_transforms
        new_denorm = copy.deepcopy(serve.eval.denormalize_action.to_dict())
        old_denorm = copy.deepcopy(released.eval.denormalize_action.to_dict())
        new_denorm.pop('norm_stats')
        old_denorm.pop('norm_stats')
        assert new_denorm == old_denorm


def test_dataset_uses_frozen_v3_contract(cfg):
    wrapper = cfg.train_dataloader.dataset
    assert wrapper.type == 'DistributedRepeatingDataset'
    assert wrapper.statistic_name == 'robolab_droid'
    dataset = wrapper.datasets
    assert dataset.type == 'ParquetDatasetV3'
    assert dataset.referenced_data_files_only is True
    assert dataset.sample_indices_path
    assert dataset.action_window_size == 16
    assert dataset.window_start_idx == 0
    assert dataset.action_key == 'action'
    assert dataset.supervise_terminal_padding is True
    assert dataset.attach_episode_video_meta is True
    assert list(dataset.episode_video_keys[0]) == [
        'observation.images.exterior_1_left',
        'observation.images.exterior_2_left'
    ]
    assert dataset.episode_video_keys[1] == 'observation.images.wrist_left'
    assert len(dataset.episode_task_keys) == 3
    relative = dataset.transforms[1]
    assert relative.type == 'RelativeActions'
    assert list(relative.mask) == [True] * 7 + [False]
    assert dataset.transforms[-1].type == 'OpenPIImageAugment'


def test_statistics_are_the_approved_base_droid_asset(cfg):
    stats = cfg.train_dataloader.dataset.dataset_statistics['robolab_droid']
    source = {
        'norm_stats': {
            'actions': stats['action'],
            'state': stats['state']
        }
    }
    encoded = json.dumps(source, indent=2).encode()
    assert hashlib.sha256(encoded).hexdigest() == BASE_DROID_STATS_SHA256
    for channel in stats.values():
        assert all(len(channel[key]) == 8 for key in ('q01', 'q99'))
        assert np.all(np.asarray(channel['q99']) >= channel['q01'])


def test_learning_rate_warms_up_then_stays_constant(cfg):
    schedule = cfg.runner.lr_scheduler
    assert schedule.schedule_style == 'openpi'
    floor = schedule.min_lr / cfg.runner.optimizer.lr
    assert floor == 1.0

    def multiplier(step):
        return LinearWarmupCosineDecayLRScheduler._openpi_lr_multiplier(
            step, schedule.warmup_steps, schedule.decay_steps, floor)

    assert multiplier(0) == pytest.approx(1 / 1001)
    assert multiplier(500) < multiplier(999) < 1.0
    assert [multiplier(step) for step in (1000, 50000, 99999)] == [1.0] * 3


def test_training_inputs_match_robolab_serving_inputs(cfg, tmp_path,
                                                      monkeypatch):
    asset = tmp_path / 'tokenizer.model'
    asset.write_bytes(b'local-test-processor')
    prompts = []

    class Recorder:

        def encode(self, text, **kwargs):
            prompts.append(text)
            return [2] + [ord(char) % 200 for char in text]

    monkeypatch.setattr(tokenizer_module, 'maybe_download',
                        lambda *a, **k: asset)
    monkeypatch.setattr(tokenizer_module.sentencepiece,
                        'SentencePieceProcessor', lambda **kwargs: Recorder())

    rng = np.random.default_rng(0)
    exterior = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    stats = copy.deepcopy(
        cfg.train_dataloader.dataset.dataset_statistics['robolab_droid'])
    low = np.asarray(stats['state']['q01'])
    high = np.asarray(stats['state']['q99'])
    state = (low + (high - low) * rng.uniform(0.1, 0.9, 8)).astype(
        np.float32).astype(np.float64)
    task = 'Put the marker in the cup'

    train_cfg = copy.deepcopy(
        cfg.train_dataloader.dataset.datasets.transforms[1:-1])
    for spec in train_cfg:
        if spec['type'] == 'ProcessPrompts':
            spec['tokenizer']['model_path'] = str(asset)
    train_transforms = [build_transform_from_cfg(spec) for spec in train_cfg]
    sample = dict(
        states=state.copy(),
        actions=np.repeat(state[None], 16, axis=0),
        action_masks=np.ones(16, dtype=np.float32),
        images=[
            exterior.transpose(2, 0, 1),
            wrist.transpose(2, 0, 1),
            np.zeros((3, 180, 320), dtype=np.uint8)
        ],
        img_masks=np.array([True, True, False]),
        task_description=task,
        stats=stats)
    for transform in train_transforms:
        sample = transform(sample)

    serve_cfg = Config.fromfile(str(SERVE)).eval.dataset
    serve_cfg.norm_stats = {'robolab_droid': stats}
    for spec in serve_cfg.transforms:
        if spec['type'] == 'ProcessPrompts':
            spec['tokenizer']['model_path'] = str(asset)
    serving = build_dataset_from_cfg(serve_cfg)
    batch, _ = serving({
        'over_shoulder_left_camera': exterior,
        'wrist_cam': wrist,
        'states': state.copy(),
        'task_description': task,
    })

    assert prompts[0] == prompts[1]
    np.testing.assert_allclose(
        sample['states'], batch['states'][0].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sample['lang_tokens'],
                                  batch['lang_tokens'][0].numpy())
    np.testing.assert_array_equal(sample['lang_masks'],
                                  batch['lang_masks'][0].numpy())
    np.testing.assert_array_equal(sample['img_masks'],
                                  batch['img_masks'][0].numpy())
    np.testing.assert_allclose(
        np.asarray(sample['images'], dtype=np.float32),
        batch['images'][0].numpy(),
        rtol=0,
        atol=1e-6)
    # Equal action and state rows: joint deltas are zero, the gripper stays
    # absolute; both use the action statistics and pad to 32 dimensions.
    q01 = np.asarray(stats['action']['q01'])
    q99 = np.asarray(stats['action']['q99'])
    target = np.zeros(8)
    target[7] = state[7]
    expected = (target - q01) / (q99 - q01 + 1e-6) * 2 - 1
    actions = np.asarray(sample['actions'])
    assert actions.shape == (16, 32)
    np.testing.assert_allclose(
        actions[:, :8], np.repeat(expected[None], 16, 0), atol=1e-6)
    assert not actions[:, 8:].any()
