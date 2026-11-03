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
"""Serve the OpenPI pi05_droid_jointpos checkpoint on RoboLab observations.

Like the RoboDojo recipe, this config uses ordinary PI05FlowMatching and
composes shared transforms. The checkpoint requires 32 internal action
dimensions, a 15-action horizon and 10 denoising updates. RoboLab exposes
seven joint positions and one gripper position, with two RGB cameras.

Set FLUXVLA_ROBOLAB_CHECKPOINT, FLUXVLA_ROBOLAB_TOKENIZER and
FLUXVLA_ROBOLAB_STATS to local assets before starting the serving runner.
Statistics use the FluxVLA schema: robolab_droid -> state/action ->
mean/std/q01/q99, taking the first eight entries of the checkpoint's
official 32-dimensional DROID statistics. Do not substitute training-set
statistics from another benchmark or change the original asset file.

BF16 autocast keeps the existing parameter containers: the standard loader
retains 37 BF16 backbone norm tensors even with torch_dtype='float32'.
keep_params_fp32 prevents a blanket BF16 cast; it does not upcast those
tensors. For a diagnostic run without autocast, disable both
model/inference_model.enable_mixed_precision_training and
themis.server.enable_mixed_precision in the loaded config.

This is an inference recipe. It defines the model service, not RoboLab's
simulation launcher, task scheduling or evaluation reporting.
"""

import os

_CHECKPOINT_PATH = os.environ.get(
    'FLUXVLA_ROBOLAB_CHECKPOINT',
    './checkpoints/pi05_droid_jointpos/model.safetensors')
_TOKENIZER_PATH = os.environ.get(
    'FLUXVLA_ROBOLAB_TOKENIZER',
    './checkpoints/pi05_droid_jointpos/paligemma_tokenizer.model')
_STATS_PATH = os.environ.get(
    'FLUXVLA_ROBOLAB_STATS',
    './checkpoints/pi05_droid_jointpos/dataset_statistics.json')
_IMAGE_KEYS = ['over_shoulder_left_camera', 'wrist_cam']
_DELTA_MASK = [True] * 7 + [False]

model = dict(
    type='PI05FlowMatching',
    openpi_fp32_flow=True,
    enable_mixed_precision_training=True,
    attention_implementation='sdpa',
    # PaliGemma image/language backbone, matching the OpenPI checkpoint.
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            _attn_implementation='eager',
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-6,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False)),
    projector=dict(type='LinearProjector', in_dim=1152, out_dim=2048),
    proj_width=1024,
    n_action_steps=15,
    num_steps=10,
    max_action_dim=32,
    ori_action_dim=8,
    state_proj=None,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    llm_expert=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=1024,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    # The serving builder loads the explicit checkpoint exactly once.
    pretrained_name_or_path=None,
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    strict_mapping=True)

inference_model = model.copy()

eval = dict(
    dataset=dict(
        type='RoboLabEvalDataset',
        img_keys=_IMAGE_KEYS,
        norm_stats=_STATS_PATH,
        unnorm_key='robolab_droid',
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
            # Tokenize the native state before padding its numeric view.
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
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_stats=_STATS_PATH,
        statistic_name='robolab_droid',
        norm_type='quantile',
        quantile_epsilon=1e-6,
        preserve_intermediate_dtype=True,
        action_dim=8,
        delta_action_mask=_DELTA_MASK,
        clip_normalized_action=False,
        normalize_gripper_action=False,
        invert_gripper_action=False))

themis = dict(
    transport=dict(
        image_keys=_IMAGE_KEYS,
        state_keys=['states'],
        unnorm_key='robolab_droid'),
    server=dict(
        runner=dict(type='FluxVLAServingRunner'),
        ckpt_path=_CHECKPOINT_PATH,
        norm_stats_path=_STATS_PATH,
        device='cuda:0',
        enable_mixed_precision=True,
        keep_params_fp32=True,
        mixed_precision_dtype='bf16',
        denormalize_context=dict(task_suite_name='robolab_droid')))
