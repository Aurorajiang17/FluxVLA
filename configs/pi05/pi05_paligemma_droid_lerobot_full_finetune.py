# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full PI0.5 fine-tuning on LeRobot v3 DROID 1.0.1 for RoboLab.

The model, prompt and image pipeline are inherited from
``pi05_paligemma_robolab_droid_lerobot_inference.py``, which serves the
checkpoints trained here: training and serving both use 16-step chunks. Do
not serve them with the released-model ``..._jointpos_*`` configs (15 steps).

The recipe follows OpenPI ``pi05_full_droid_finetune`` (pinned reference
xuningy/openpi@aa6420561529593114160d05e5ad155792b272f3): ``pi05_base``
initialization, Beta(1.5, 1) flow time, 32-dimensional loss, AdamW 5e-5 with
betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, gradient clipping 1.0, 1,000
warmup steps followed by a constant learning rate, EMA 0.99, global batch 256
and 100k updates. The runner rejects any other global batch, which is
GPUs x per_device_batch_size x grad_accumulation_steps: with 4 GPUs use
``runner.grad_accumulation_steps=2``. This is an approximate open-source
recipe, not the training log of the released checkpoint.

Data are read from the LeRobot v3 files referenced by ``meta/episodes``. The
only sample starts are the frozen global indices in FLUXVLA_DROID_SAMPLE_STARTS
(successful episodes, idle-filtered, validation episodes excluded). Actions
are 16 consecutive ``action`` rows; past the episode end the last action is
repeated and still supervised. The seven joints are relative to the current
state and the gripper stays absolute. One exterior camera and one of three
instructions are chosen per episode and epoch; empty instructions are kept.
Statistics are the public base DROID statistics (source SHA256
508e75bb8f28f86ad8998a2254d2076fdd42caf26d8a89baecd5de0f628e2280), saved
with each run as ``robolab_droid``.

Resume restores weights, EMA, optimizer, scheduler, step and the data
position, and fails if any of them is missing or cannot be loaded. The data
stream continues after the samples already trained on in the current epoch;
it requires the same GPU count, batch, accumulation, workers and data.
Checkpoints are saved every 5,000 steps; the newest two keep full training
state and every 10,000-step checkpoint keeps its weights for evaluation.
"""

import os

_base_ = ['./pi05_paligemma_robolab_droid_lerobot_inference.py']

_DATA_ROOT = os.environ.get('FLUXVLA_DROID_ROOT', './datasets/droid_1.0.1')
_SAMPLE_STARTS = os.environ.get(
    'FLUXVLA_DROID_SAMPLE_STARTS',
    './datasets/droid_1.0.1_index/valid_sample_indices_h16.npy')
_PI05_BASE = os.environ.get('FLUXVLA_PI05_BASE',
                            './checkpoints/pi05_base/model.safetensors')
_TOKENIZER = os.environ.get('FLUXVLA_PI05_TOKENIZER',
                            './checkpoints/pi05_base/tokenizer.model')

train_seed = 42
allow_eval_after_train = False

_STATISTIC_NAME = 'robolab_droid'
_EXTERIOR_CAMERAS = [
    'observation.images.exterior_1_left',
    'observation.images.exterior_2_left',
]
_WRIST_CAMERA = 'observation.images.wrist_left'
_INSTRUCTIONS = [
    'language_instruction',
    'language_instruction_2',
    'language_instruction_3',
]
_JOINT_DELTA_MASK = [True] * 7 + [False]

_DROID_STATS = {
    'robolab_droid': {
        'state': {
            'mean': [
                0.016363732516765594,
                0.2619902193546295,
                -0.016933832317590714,
                -2.027378797531128,
                -0.032851383090019226,
                2.3424031734466553,
                0.08234210312366486,
                0.3888995349407196,
            ],
            'std': [
                0.30887290835380554,
                0.4825698137283325,
                0.27116459608078003,
                0.4830065071582794,
                0.5358471274375916,
                0.4547894597053528,
                0.7465448379516602,
                0.4062422513961792,
            ],
            'q01': [
                -0.8321998119354248,
                -0.8423057794570923,
                -0.8551784753799438,
                -2.773015260696411,
                -1.8426181077957153,
                1.1692570447921753,
                -2.051669120788574,
                0.0,
            ],
            'q99': [
                0.9102187752723694,
                1.3836115598678589,
                0.6993954181671143,
                -0.4531737267971039,
                1.7291260957717896,
                3.4688961505889893,
                2.2018003463745117,
                0.9909999966621399,
            ],
        },
        'action': {
            'mean': [
                -0.005160212982445955,
                0.013146234676241875,
                0.002338092541322112,
                0.028497571125626564,
                0.0015346353175118566,
                -0.0015720161609351635,
                0.0025643655098974705,
                0.43498775362968445,
            ],
            'std': [
                0.1532931625843048,
                0.30146104097366333,
                0.14914393424987793,
                0.294704794883728,
                0.22254551947116852,
                0.24063511192798615,
                0.2638004422187805,
                0.44128793478012085,
            ],
            'q01': [
                -0.45159998536109924,
                -0.7979999780654907,
                -0.438400000333786,
                -0.9088000059127808,
                -0.6340000033378601,
                -0.6327999830245972,
                -0.7516000270843506,
                0.0,
            ],
            'q99': [
                0.43880000710487366,
                0.7644000053405762,
                0.4431999921798706,
                0.7860000133514404,
                0.6380000114440918,
                0.6567999720573425,
                0.724399983882904,
                0.998199999332428,
            ],
        },
    },
}

# Only the training fields differ from the serving model.
model = dict(
    pretrained_name_or_path=_PI05_BASE,
    n_action_steps=16,
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    loss_action_dim=32,
    freeze_llm_backbone=False,
    freeze_vision_backbone=False)

train_dataloader = dict(
    # 32 samples/GPU x 8 GPUs x 1 accumulation step = global batch 256.
    per_device_batch_size=32,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=train_seed,
        shuffle=True,
        reshuffle_each_epoch=True,
        statistic_keys=['observation.state', 'action'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=_DROID_STATS,
        datasets=dict(
            type='ParquetDatasetV3',
            data_root_path=_DATA_ROOT,
            action_window_size=16,
            action_key='action',
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            supervise_terminal_padding=True,
            referenced_data_files_only=True,
            sample_indices_path=_SAMPLE_STARTS,
            attach_episode_video_meta=True,
            episode_video_keys=[_EXTERIOR_CAMERAS, _WRIST_CAMERA],
            episode_task_keys=_INSTRUCTIONS,
            selection_seed=train_seed,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',  # 7 joints + gripper
                        'timestamp',
                        'actions',  # 16 x 8 joint/gripper commands
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    # Replaced per sample by the chosen camera pair.
                    video_keys=[_EXTERIOR_CAMERAS[0], _WRIST_CAMERA],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    # Third PI0.5 image slot: zeros with a False mask.
                    num_padding_imgs=1,
                    video_backend='pyav'),
                dict(
                    type='RelativeActions',
                    mask=_JOINT_DELTA_MASK,
                    state_key='states',
                    action_key='actions'),
                dict(
                    type='NormalizeStatesAndActions',
                    state_key='state',
                    action_key='action',
                    norm_type='quantile',
                    clip_norm=False,
                    normalization_epsilon=1e-6,
                    output_dtype='float32'),
                # Tokenize the native 8D state before numeric padding.
                dict(
                    type='PreparePromptWithState',
                    task_key='task_description',
                    lowercase_task_description=False,
                    add_action_prefix=True),
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PaligemmaTokenizer',
                        model_path=_TOKENIZER,
                        preformatted=True)),
                dict(type='PadStatesAndActions', model_action_dim=32),
                dict(
                    type='ResizeImagesWithPad',
                    height=224,
                    width=224,
                    backend='pil'),
                dict(type='SimpleNormalizeImages'),
                # Crop/rotate only the exterior camera, as OpenPI does.
                dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
            ])))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    grad_accumulation_steps=1,
    deterministic_algorithms=True,
    ema_decay=0.99,
    seed=train_seed,
    strict_training_state=True,
    expected_global_batch_size=256,
    resume_data_progress=True,
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True),
    max_grad_norm=1.0,
    save_epoch_interval=1,
    # OpenPI saves every 5k steps and keeps every 10k-step checkpoint.
    save_iter_interval=5000,
    max_keep_ckpts=2,
    milestone_interval=10000,
    sharding_strategy='full-shard',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',  # (B, 32) normalized state, zero padded
            'images',  # (B, 9, 224, 224): exterior, wrist, padding
            'img_masks',  # (B, 3)
            'lang_tokens',  # (B, 200)
            'lang_masks',  # (B, 200)
            'actions',  # (B, 16, 32) normalized and padded
            'action_masks',  # (B, 16)
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs',
        window_size=1),
    # OpenPI warmup, then cosine decay from 5e-5 to 5e-5, i.e. constant.
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=1000000,
        min_lr=5e-5),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)
