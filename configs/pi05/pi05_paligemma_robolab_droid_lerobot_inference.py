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
"""Serve PI0.5 checkpoints trained with the LeRobot DROID fine-tune recipe.

Use this config only for checkpoints produced by
``pi05_paligemma_droid_lerobot_full_finetune.py``. The released
``pi05_droid_jointpos`` checkpoint keeps its own 15-step configs. Relative to
the process-stream RoboLab recipe it inherits, only these settings differ:

* The model returns 16-step chunks, the training horizon of the OpenPI
  ``pi05_full_droid_finetune`` recipe. Chunks are not truncated here.
* FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT, FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER and
  FLUXVLA_ROBOLAB_DROID_FT_STATS select the trained weights, the PaliGemma
  tokenizer and the statistics. Without FLUXVLA_ROBOLAB_DROID_FT_STATS the
  server uses ``dataset_statistics.json`` of the training run, next to the
  ``checkpoints`` directory, and refuses to start if it is missing.

How many returned actions run before the next observation is an evaluation
setting (FluxThemis ``execute_horizon``; RoboLab ``open_loop_horizon``), not a
model setting. Evaluate these checkpoints executing 15 of the 16 actions, the
replanning interval of the released pi05 baseline at RoboLab's 15 Hz control.
The stock FluxThemis ``configs/robolab/droid.py`` sets ``execute_horizon=None``
and would execute all 16, so point ``simulator_config`` in the process launch
file at a RoboLab config that inherits it with ``execute_horizon=15``.
"""

import os

_base_ = './pi05_paligemma_robolab_droid_jointpos_process_inference.py'

_CHECKPOINT_PATH = os.environ.get(
    'FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT',
    './work_dirs/pi05_paligemma_droid_lerobot_full_finetune/checkpoints/'
    'latest-checkpoint.safetensors')
_TOKENIZER_PATH = os.environ.get('FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER',
                                 './checkpoints/pi05_base/tokenizer.model')
_STATS_PATH = os.environ.get('FLUXVLA_ROBOLAB_DROID_FT_STATS')
_IMAGE_KEYS = ['over_shoulder_left_camera', 'wrist_cam']
_ACTION_STEPS = 16

model = dict(n_action_steps=_ACTION_STEPS)
inference_model = dict(n_action_steps=_ACTION_STEPS)

# Same input chain as the released-model recipe; only the tokenizer path
# differs. A list replaces the inherited one, so it is repeated in full.
eval = dict(
    dataset=dict(
        norm_stats=_STATS_PATH,
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=_IMAGE_KEYS,
                num_padding_imgs=1),
            dict(
                type='StateFromInputs',
                state_dim=None,
                stat_key='state',
                norm_type='quantile',
                clip_norm=False,
                preserve_input_dtype=True),
            dict(
                type='PreparePromptWithState',
                task_key='task_description',
                lowercase_task_description=False,
                add_action_prefix=True),
            dict(
                type='ProcessPrompts',
                tokenizer=dict(
                    type='PaligemmaTokenizer',
                    model_path=_TOKENIZER_PATH,
                    preformatted=True),
                max_len=200),
            dict(type='PadStatesAndActions', model_action_dim=32),
            dict(
                type='TransformImage',
                input_sizes=[[3, 224, 224]] * 3,
                image_resize_strategy='letterbox',
                letterbox_fill=[0, 0, 0],
                letterbox_pad_position='center',
                letterbox_resize_rounding='floor',
                means=[[127.5, 127.5, 127.5]] * 3,
                stds=[[127.5, 127.5, 127.5]] * 3),
            dict(
                type='PackModelInputs',
                key_mapping=dict(
                    images='pixel_values',
                    img_masks='img_masks',
                    lang_tokens='lang_tokens',
                    lang_masks='lang_masks',
                    states='states'),
                dtypes=dict(
                    images='float32',
                    img_masks='bool',
                    lang_tokens='int64',
                    lang_masks='bool',
                    states='float32'),
                add_batch_dim=True),
        ]),
    denormalize_action=dict(norm_stats=_STATS_PATH))

themis = dict(
    server=dict(ckpt_path=_CHECKPOINT_PATH, norm_stats_path=_STATS_PATH))
