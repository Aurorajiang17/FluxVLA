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
"""Configurable online RoboLab observations for model input pipelines."""

import copy
import json
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import numpy as np

from fluxvla.engines.utils.root import DATASETS

__all__ = ['RoboLabEvalDataset']


@DATASETS.register_module()
class RoboLabEvalDataset:
    """Validate an observation, then run its configured input transforms.

    Images are named uint8 HWC RGB arrays; state is a finite, nonempty real
    vector. Camera names/counts, state width and model batch fields are not
    fixed here. The final transform owns model-specific batch packing.
    Transforms should produce CPU inputs; the serving runner owns device
    placement. This adapter performs no normalization or dtype conversion.

    Args:
        img_keys: Required camera keys in the online observation.
        transforms: Nonempty sequence of transform configurations.
        state_key: Observation field containing the raw robot state.
        task_key: Observation field containing the task instruction.
        norm_stats: Optional statistics mapping or local JSON file path.
        unnorm_key: Explicit statistics entry, required with norm_stats.

    Each call uses independent observation/statistics copies. After a
    successful call, last_raw_state contains a separate copy for the
    existing serving runner's action decoder. It is cleared before each
    request, including requests that fail. This is a serial request
    adapter, not an episode-history or concurrent-request store.
    """

    def __init__(self,
                 *,
                 img_keys: Sequence[str],
                 transforms: Sequence[Mapping],
                 state_key: str = 'states',
                 task_key: str = 'task_description',
                 norm_stats: Any = None,
                 unnorm_key: str | None = None) -> None:
        if isinstance(img_keys,
                      (str, bytes)) or not isinstance(img_keys, Sequence):
            raise TypeError('img_keys must be a sequence of camera names')
        if not img_keys:
            raise ValueError('img_keys must not be empty')
        if any(not isinstance(key, str) for key in img_keys):
            raise TypeError('img_keys must contain strings')
        if any(not key
               for key in img_keys) or len(set(img_keys)) != len(img_keys):
            raise ValueError('img_keys must contain nonempty unique names')
        for label, key in [('state_key', state_key), ('task_key', task_key)]:
            if not isinstance(key, str) or not key:
                raise ValueError(f'{label} must be a nonempty string')
        if isinstance(transforms,
                      (str, bytes)) or not isinstance(transforms, Sequence):
            raise TypeError('transforms must be a sequence of configurations')
        if not transforms:
            raise ValueError('transforms must include a batch packing step')
        if any(not isinstance(cfg, Mapping) for cfg in transforms):
            raise TypeError('transforms must contain mapping configurations')
        if unnorm_key is not None and (not isinstance(unnorm_key, str)
                                       or not unnorm_key):
            raise ValueError('unnorm_key must be a nonempty string or None')

        if isinstance(norm_stats, (str, PathLike)):
            with open(norm_stats, encoding='utf-8') as stream:
                norm_stats = json.load(stream)
        if norm_stats is not None:
            if not isinstance(norm_stats, Mapping):
                raise TypeError('norm_stats must be a mapping or JSON path')
            if unnorm_key is None:
                raise ValueError('norm_stats requires an explicit unnorm_key')
            if unnorm_key not in norm_stats:
                raise KeyError(f'Unknown unnorm_key: {unnorm_key!r}')
            if not isinstance(norm_stats[unnorm_key], Mapping):
                raise TypeError(f'Statistics {unnorm_key!r} must be a mapping')

        from fluxvla.engines import build_transform_from_cfg

        self.img_keys = tuple(img_keys)
        self.state_key = state_key
        self.task_key = task_key
        self.unnorm_key = unnorm_key
        self.norm_stats = copy.deepcopy(norm_stats)
        self.transforms = [
            build_transform_from_cfg(copy.deepcopy(dict(cfg)))
            for cfg in transforms
        ]
        self.last_raw_state = None

    def _validate_observation(self, inputs: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(inputs, Mapping):
            raise TypeError('RoboLab observation must be a mapping')
        for key in self.img_keys:
            if key not in inputs:
                raise KeyError(f'Missing image key: {key!r}')
            try:
                image = np.asarray(inputs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Image {key!r} must be an RGB array') from exc
            if image.dtype != np.uint8:
                raise TypeError(f'Image {key!r} must have uint8 dtype')
            if (image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] == 0
                    or image.shape[1] == 0):
                raise ValueError(f'Image {key!r} must be nonempty [H, W, 3]')
        if self.state_key not in inputs:
            raise KeyError(f'Missing state key: {self.state_key!r}')
        try:
            state = np.asarray(inputs[self.state_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'State {self.state_key!r} must be a real vector') from exc
        if (state.ndim != 1 or state.size == 0 or state.dtype.kind not in 'iuf'
                or not np.isfinite(state).all()):
            raise ValueError(
                f'State {self.state_key!r} must be a nonempty finite real '
                '1-D array')
        if self.task_key not in inputs:
            raise KeyError(f'Missing task key: {self.task_key!r}')
        if not isinstance(inputs[self.task_key], str):
            raise TypeError(f'Task {self.task_key!r} must be a string')
        return state

    def __call__(self, inputs: Mapping[str, Any]) -> tuple[dict, None]:
        """Return the configured model batch and no replay image."""
        self.last_raw_state = None
        raw_state = self._validate_observation(inputs).copy()
        data = copy.deepcopy(dict(inputs))
        # Statistics come from configuration, never from the request payload.
        data.pop('norm_stats', None)
        if self.norm_stats is not None:
            data['norm_stats'] = copy.deepcopy(
                self.norm_stats[self.unnorm_key])
        for index, transform in enumerate(self.transforms):
            data = transform(data)
            if not isinstance(data, Mapping):
                raise TypeError(
                    f'RoboLab transform {index} ({type(transform).__name__}) '
                    'must return a mapping')
        batch = dict(data)
        self.last_raw_state = raw_state
        return batch, None
