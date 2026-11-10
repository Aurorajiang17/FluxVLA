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
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.engines import DATASETS, build_transform_from_cfg


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_parquet_records(paths: List[Path]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    dataset = load_dataset(
        'parquet', data_files=[str(path) for path in paths], split='train')
    return [dict(record) for record in dataset]


def _load_episode_records(meta_root: Path) -> List[Dict[str, Any]]:
    legacy_path = meta_root / 'episodes.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    episodes_dir = meta_root / 'episodes'
    parquet_files = sorted(episodes_dir.rglob('*.parquet'))
    if parquet_files:
        return _load_parquet_records(parquet_files)

    raise FileNotFoundError(f'Episodes metadata not found under {meta_root}')


def _load_stats_records(meta_root: Path) -> List[Dict[str, Any]]:
    v3_stats_path = meta_root / 'stats.json'
    if v3_stats_path.exists():
        return [{'stats': _load_json(v3_stats_path)}]

    legacy_path = meta_root / 'episodes_stats.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    raise FileNotFoundError(f'Statistics metadata not found under {meta_root}')


def _extract_task_text(record: Dict[str, Any]) -> str:
    for key in ('task', 'tasks', '__index_level_0__'):
        value = record.get(key)
        if value is not None:
            return str(value)

    for key, value in record.items():
        if key == 'task_index' or value is None:
            continue
        if isinstance(value, str):
            return value

    for key, value in record.items():
        if key != 'task_index' and value is not None:
            return str(value)

    return ''


def _load_task_mapping(meta_root: Path) -> Dict[int, str]:
    v3_path = meta_root / 'tasks.parquet'
    if v3_path.exists():
        records = _load_parquet_records([v3_path])
    else:
        legacy_path = meta_root / 'tasks.jsonl'
        if not legacy_path.exists():
            raise FileNotFoundError(
                f'Tasks metadata not found under {meta_root}')
        records = _load_jsonl_records(legacy_path)

    tasks = {}
    for fallback_index, record in enumerate(records):
        task_index = record.get('task_index', fallback_index)
        if isinstance(task_index, np.ndarray) and task_index.ndim == 0:
            task_index = task_index.item()
        if isinstance(task_index, torch.Tensor) and task_index.numel() == 1:
            task_index = task_index.item()
        tasks[int(task_index)] = _extract_task_text(record)
    return tasks


_EPISODE_CORE_COLUMNS = ('episode_index', 'dataset_from_index',
                         'dataset_to_index', 'data/chunk_index',
                         'data/file_index')


def _load_episode_columns(meta_root: Path) -> Dict[str, np.ndarray]:
    """Read compact v3 episode location columns, skipping per-episode stats.

    Returns one array per column, ordered by ``episode_index``.
    """
    paths = sorted((meta_root / 'episodes').rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError(
            f'v3 episodes metadata not found under {meta_root}')
    names = pq.read_schema(paths[0]).names
    missing = [name for name in _EPISODE_CORE_COLUMNS if name not in names]
    if missing:
        raise KeyError(f'Episodes metadata lacks columns: {missing}')
    columns = [
        name for name in names
        if name in _EPISODE_CORE_COLUMNS or name == 'length' or (
            name.startswith('videos/') and name.rsplit('/', 1)[-1] in
            ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'))
    ]
    parts = {name: [] for name in columns}
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for name in columns:
            parts[name].append(table.column(name).to_numpy())
    result = {name: np.concatenate(values) for name, values in parts.items()}
    order = np.argsort(result['episode_index'], kind='stable')
    result = {name: values[order] for name, values in result.items()}
    if np.any(np.diff(result['episode_index']) <= 0):
        raise ValueError('Duplicate episode_index in episodes metadata')
    return result


def _scalar_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if not isinstance(value, str):
        raise TypeError(f'Expected a text column value, got {type(value)}')
    return value


@DATASETS.register_module()
class ParquetDatasetV3(ParquetDataset):
    """ParquetDataset variant for LeRobot v3 metadata layouts.

    This subclass keeps the original ``ParquetDataset`` unchanged while adding
    support for v3 metadata files such as ``tasks.parquet``,
    ``meta/episodes/*.parquet`` and ``stats.json``.
    """

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 frame_window_size: int = 1,
                 frame_sample_stride: int = 1,
                 expose_index: bool = False,
                 supervise_terminal_padding: bool = False,
                 action_dtype: Optional[str] = 'float32',
                 referenced_data_files_only: bool = False,
                 data_episodes: Optional[Sequence] = None,
                 sample_indices_path: Optional[str] = None,
                 attach_episode_video_meta: bool = False,
                 episode_video_keys: Optional[Sequence] = None,
                 episode_task_keys: Optional[Sequence] = None,
                 selection_seed: int = 0) -> None:
        """Initialize a parquet dataset backed by LeRobot v3 metadata.

        Args:
            data_root_path (Union[str, List[str]]): One dataset root, or a list
                of dataset roots, each containing ``meta`` and ``data``.
            transforms (List[Dict]): Transform config dictionaries applied in
                order after sample assembly.
            action_window_size (int): Number of future actions to return.
            action_key (str): Dataset key used as the action/state source.
            use_delta (bool): Whether actions are converted to frame-to-frame
                deltas.
            statistic_name (str): Statistics entry read from
                ``dataset_statistics``.
            window_start_idx (int): First offset used when building the action
                window.
            frame_window_size (int): Number of timestamps to expose for
                frame-sequence consumers.
            frame_sample_stride (int): Stride (in dataset rows) between sampled
                video frames. Defaults to 1. Increase this when the sampled
                frames should span a longer temporal window.
            expose_index (bool): Whether to expose the concatenated row index
                to transforms for offline sample weighting.
            supervise_terminal_padding (bool): Keep repeated final actions
                past the episode end valid in ``action_masks`` (OpenPI and
                LeRobot supervise them). Defaults to False.
            action_dtype (str, optional): NumPy dtype of assembled action
                windows; None keeps the source dtype. Defaults to float32.
            referenced_data_files_only (bool): Load only the data files that
                ``meta/episodes`` references (optionally restricted to
                ``data_episodes``) instead of every file under ``data/``, and
                check that their rows form one contiguous range of the global
                ``index`` column. Requires one data root. Episode metadata is
                then kept as compact location columns and ``episodes`` stays
                empty. Defaults to False.
            data_episodes (Sequence[int], optional): Episodes whose data files
                are loaded in referenced mode. None selects all episodes.
            sample_indices_path (str, optional): ``.npy`` file of strictly
                increasing global ``index`` values used as the only sample
                starts, in referenced mode. Starts are never resampled; the
                action window takes consecutive rows of the same episode and
                repeats the last action past its end. ``use_delta`` and frame
                windows are not supported in this mode.
            attach_episode_video_meta (bool): Add the episode's v3 video
                chunk/file indices and ``from_timestamp`` offsets as
                ``lerobot_episode_meta`` for :class:`ProcessParquetInputs`.
                Requires frozen sample starts.
            episode_video_keys (Sequence, optional): Video keys to decode, as
                ``lerobot_video_keys``. A nested sequence is a group from which
                one key is chosen per episode and sample epoch.
            episode_task_keys (Sequence[str], optional): Text columns from
                which one prompt is chosen per episode and sample epoch.
            selection_seed (int): Seed of the episode-level choices. The
                choice value is the first eight bytes of
                ``sha256(f'{seed}\\0{epoch}\\0{episode}')``; groups consume
                it in mixed radix, video groups first, then task keys.
        """
        Dataset.__init__(self)
        self.action_window_size = action_window_size
        if isinstance(data_root_path, str):
            data_root_path = [data_root_path]
        self.data_root_path = data_root_path
        frozen_starts = sample_indices_path is not None
        if referenced_data_files_only and len(data_root_path) != 1:
            raise ValueError('referenced_data_files_only needs one data root')
        if data_episodes is not None and not referenced_data_files_only:
            raise ValueError('data_episodes requires referenced data files')
        if frozen_starts and not referenced_data_files_only:
            raise ValueError('sample_indices_path requires referenced data')
        if frozen_starts and (use_delta or frame_window_size != 1):
            raise ValueError('Frozen starts do not support use_delta or '
                             'frame windows')
        selection_requested = (
            attach_episode_video_meta or episode_video_keys is not None
            or episode_task_keys is not None)
        if selection_requested and not frozen_starts:
            raise ValueError('Episode video/task options require '
                             'sample_indices_path')

        meta_root = [os.path.join(path, 'meta') for path in data_root_path]
        data_root = [os.path.join(path, 'data') for path in data_root_path]

        all_stats = []
        all_tasks = []
        all_episodes = []
        episodes_by_dataset = []
        info_list = []
        self._episode_columns = None

        for root in meta_root:
            info_path = os.path.join(root, 'info.json')
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            info = _load_json(Path(info_path))
            if 'features' in info:
                for feature in info['features'].values():
                    if isinstance(feature.get('shape'), list):
                        feature['shape'] = tuple(feature['shape'])
            info_list.append(info)

            meta_path = Path(root)
            all_stats.extend(_load_stats_records(meta_path))
            all_tasks.append(_load_task_mapping(meta_path))

            if referenced_data_files_only:
                self._episode_columns = _load_episode_columns(meta_path)
                episode_records = []
            else:
                episode_records = _load_episode_records(meta_path)
            episodes_by_dataset.append(episode_records)
            all_episodes.extend(episode_records)

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.episodes = all_episodes
        self.episodes_by_dataset = episodes_by_dataset

        datasets = []
        dataset_sizes = []
        self._global_index_offset = 0
        for root in data_root:
            if referenced_data_files_only:
                hf_dataset = self._load_referenced_data(
                    Path(data_root_path[0]), info_list[0], data_episodes)
                datasets.append(hf_dataset)
                dataset_sizes.append(len(hf_dataset))
                continue
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = concatenate_datasets(datasets)
        self.full_length = len(self.dataset)
        self.sample_indices = np.arange(self.full_length, dtype=np.int64)
        self.effective_length = self.full_length
        self.transforms = list()
        self.action_key = action_key
        self.use_delta = use_delta
        self.statistic_name = statistic_name
        self.window_start_idx = window_start_idx
        self.frame_window_size = frame_window_size
        self.frame_sample_stride = frame_sample_stride
        self.expose_index = expose_index
        self.supervise_terminal_padding = supervise_terminal_padding
        self.action_dtype = (None if action_dtype is None else
                             np.dtype(action_dtype))
        self.frozen_starts = frozen_starts
        self.attach_episode_video_meta = attach_episode_video_meta
        self.episode_video_keys = (None if episode_video_keys is None else [
            key if isinstance(key, str) else list(key)
            for key in episode_video_keys
        ])
        self.episode_task_keys = (None if episode_task_keys is None else
                                  list(episode_task_keys))
        self.selection_seed = int(selection_seed)
        self._sample_epoch = 0
        if self.episode_video_keys is not None:
            for key in self.episode_video_keys:
                if not isinstance(key, str) and (not key or not all(
                        isinstance(item, str) for item in key)):
                    raise ValueError('Video key groups must list strings')
        if self.episode_task_keys is not None and (
                not self.episode_task_keys
                or any(not isinstance(key, str)
                       for key in self.episode_task_keys)):
            raise ValueError('episode_task_keys must list column names')
        if frozen_starts:
            self._load_frozen_starts(sample_indices_path)
            self._window_view = self.dataset.select_columns(
                [self.action_key, 'episode_index'])
        for transform in transforms:
            self.transforms.append(build_transform_from_cfg(transform))

    def _load_referenced_data(self, dataset_root: Path, info: Dict,
                              data_episodes: Optional[Sequence]):
        """Load referenced data files and bind rows to global indices."""
        columns = self._episode_columns
        episode_ids = columns['episode_index']
        if data_episodes is None:
            rows = np.arange(len(episode_ids))
        else:
            wanted = np.unique(np.asarray(list(data_episodes), dtype=np.int64))
            rows = np.searchsorted(episode_ids, wanted)
            if (wanted.size == 0 or np.any(rows >= len(episode_ids)) or np.any(
                    episode_ids[np.minimum(rows,
                                           len(episode_ids) - 1)] != wanted)):
                raise KeyError('data_episodes contains unknown episodes')
        file_keys = sorted({(int(columns['data/chunk_index'][row]),
                             int(columns['data/file_index'][row]))
                            for row in rows})
        paths = [
            dataset_root /
            info['data_path'].format(chunk_index=chunk, file_index=file)
            for chunk, file in file_keys
        ]
        starts = []
        expected = None
        for path in paths:
            index = pq.read_table(
                path, columns=['index']).column('index').to_numpy()
            if index.size == 0:
                raise ValueError(f'Empty data file: {path}')
            first = int(index[0]) if expected is None else expected
            if not np.array_equal(index, np.arange(first, first + index.size)):
                raise ValueError('Referenced data files do not form one '
                                 f'contiguous global index range: {path}')
            starts.append(first)
            expected = first + index.size
        offset = starts[0]
        lower = columns['dataset_from_index'][rows]
        upper = columns['dataset_to_index'][rows]
        if np.any(lower < offset) or np.any(upper > expected):
            raise ValueError('Selected episodes exceed loaded data files')
        hf_dataset = load_dataset(
            'parquet', data_files=[str(path) for path in paths], split='train')
        if len(hf_dataset) != expected - offset:
            raise ValueError('Loaded row count differs from data files')
        for start in starts + [expected - 1]:
            if int(hf_dataset[start - offset]['index']) != start:
                raise ValueError('Loaded rows are not in global index order')
        self._global_index_offset = offset
        self._selected_episode_rows = rows
        return hf_dataset

    def _load_frozen_starts(self, path: str) -> None:
        starts = np.load(path, mmap_mode='r', allow_pickle=False)
        if (starts.ndim != 1 or starts.dtype.kind not in 'iu'
                or starts.size == 0):
            raise ValueError('Sample starts must be a nonempty integer vector')
        if not np.all(np.diff(starts.astype(np.int64)) > 0):
            raise ValueError('Sample starts must be strictly increasing')
        columns = self._episode_columns
        rows = self._selected_episode_rows
        lower = columns['dataset_from_index'][rows]
        upper = columns['dataset_to_index'][rows]
        covered = (np.searchsorted(starts, upper) -
                   np.searchsorted(starts, lower)).sum()
        if covered != starts.size:
            raise ValueError('Sample starts fall outside selected episodes')
        self.sample_indices = starts
        self.effective_length = int(starts.size)

    def set_sample_epoch(self, epoch: int) -> None:
        """Set the epoch used by episode-level video/task choices."""
        self._sample_epoch = int(epoch)

    def _episode_row(self, episode_index: int) -> int:
        episode_ids = self._episode_columns['episode_index']
        row = int(np.searchsorted(episode_ids, episode_index))
        if row >= len(episode_ids) or episode_ids[row] != episode_index:
            raise KeyError(f'Unknown episode {episode_index}')
        return row

    def _episode_meta(self, row: int) -> Dict[str, Any]:
        return {
            name: values[row].item()
            for name, values in self._episode_columns.items()
        }

    def _episode_choices(self, episode_index: int, data: Dict) -> Dict:
        digest = hashlib.sha256(f'{self.selection_seed}\0'
                                f'{self._sample_epoch}\0'
                                f'{episode_index}'.encode()).digest()
        choice = int.from_bytes(digest[:8], 'big')
        chosen = {}
        if self.episode_video_keys is not None:
            keys = []
            for key in self.episode_video_keys:
                if isinstance(key, str):
                    keys.append(key)
                else:
                    keys.append(key[choice % len(key)])
                    choice //= len(key)
            chosen['lerobot_video_keys'] = keys
        if self.episode_task_keys is not None:
            column = self.episode_task_keys[choice %
                                            len(self.episode_task_keys)]
            chosen['task_description'] = _scalar_text(data[column])
            chosen['task_key'] = column
        return chosen

    def _getitem_frozen(self, index: int, dataset_statistics) -> Dict:
        if not 0 <= index < self.effective_length:
            raise IndexError(index)
        global_index = int(self.sample_indices[index])
        position = global_index - self._global_index_offset
        data = self.dataset[position]
        if int(data['index']) != global_index:
            raise ValueError('Sample row does not match its global index')
        episode_index = int(data['episode_index'])
        row = self._episode_row(episode_index)
        end = (
            int(self._episode_columns['dataset_to_index'][row]) -
            self._global_index_offset)
        first = position + self.window_start_idx
        stop = min(first + self.action_window_size, end)
        if first >= stop:
            raise ValueError('Sample start has no action in its episode')
        window = self._window_view[first:stop]
        if any(
                int(value) != episode_index
                for value in window['episode_index']):
            raise ValueError('Action window crosses an episode boundary')
        actions = list(window[self.action_key])
        valid = len(actions)
        padding = self.action_window_size - valid
        actions.extend([actions[-1]] * padding)
        pad_mask = 1 if self.supervise_terminal_padding else 0
        data['info'] = self.info[0]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=self.action_dtype)
        data['action_masks'] = np.array(
            [1] * valid + [pad_mask] * padding, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(global_index, dtype=np.int64)
        data['task_description'] = self._resolve_task_description(0, data)
        data.update(self._episode_choices(episode_index, data))
        if self.attach_episode_video_meta:
            data['lerobot_episode_meta'] = self._episode_meta(row)
        data['data_root'] = self.data_root_path[0]
        for transform in self.transforms:
            data = transform(data)
        return data

    def _resolve_task_description(self, dataset_idx: int, data: Dict) -> str:
        raw_task = data.get('task')
        if isinstance(raw_task, str):
            return raw_task
        if isinstance(raw_task, (list, tuple)) and len(raw_task) == 1:
            only_value = raw_task[0]
            if isinstance(only_value, str):
                return only_value
            if isinstance(only_value, (int, np.integer)):
                return self.tasks[dataset_idx].get(int(only_value), '')
        if isinstance(raw_task, np.ndarray) and raw_task.ndim == 0:
            raw_task = raw_task.item()
        if isinstance(raw_task, torch.Tensor) and raw_task.numel() == 1:
            raw_task = raw_task.item()
        if isinstance(raw_task, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task), '')

        raw_task_index = data.get('task_index')
        if isinstance(raw_task_index, np.ndarray) and raw_task_index.ndim == 0:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index,
                      torch.Tensor) and raw_task_index.numel() == 1:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task_index), '')

        return ''

    def __getitem__(self, index, dataset_statistics):
        """Build one transformed sample from a parquet row.

        Args:
            index (int): Dataset row index.
            dataset_statistics (Dict): Statistics mapping supplied by the
                runner or collator.

        Returns:
            Dict: Sample dictionary containing task text, actions, masks,
            metadata, and transform outputs.
        """
        if self.frozen_starts:
            return self._getitem_frozen(index, dataset_statistics)
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        while True:
            if index == len(self.dataset) - 1:
                needs_resample = True
            else:
                next_data = self.dataset[index + 1]
                next_task = self._resolve_task_description(
                    dataset_idx, next_data)
                needs_resample = (
                    data['episode_index'] != next_data['episode_index']
                    or self._get_dataset_index(index + 1) != dataset_idx
                    or next_task in ('empty', 'static'))

            if not needs_resample:
                break
            index = self._rand_another()
            data = self.dataset[index]
            dataset_idx = self._get_dataset_index(index)
        actions = list()
        action_masks = list()
        window_idx = self.window_start_idx
        while len(actions) < self.action_window_size:
            future_idx = index + window_idx
            future_in_range = future_idx < len(self.dataset)
            if future_in_range:
                future_data = self.dataset[future_idx]
                future_task = self._resolve_task_description(
                    dataset_idx, future_data)
                future_dataset_idx = self._get_dataset_index(future_idx)
            else:
                future_data = None
                future_task = ''
                future_dataset_idx = None

            valid_future = (
                future_in_range
                and data['episode_index'] == future_data['episode_index']
                and future_dataset_idx == dataset_idx
                and future_task not in ('empty', 'static'))
            if valid_future:
                if self.use_delta:
                    actions.append(
                        np.array(self.dataset[future_idx][self.action_key]) -
                        np.array(self.dataset[index + window_idx -
                                              1][self.action_key]).tolist())
                else:
                    actions.append(self.dataset[future_idx][self.action_key])
                action_masks.append(1)
            elif not future_in_range or future_task == 'empty':
                pad_mask = (1 if self.supervise_terminal_padding
                            and not future_in_range else 0)
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(pad_mask)
                break
            elif future_task == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(
                    1 if self.supervise_terminal_padding else 0)
            window_idx += 1

        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=self.action_dtype)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(index, dtype=np.int64)
        data['task_description'] = self._resolve_task_description(
            dataset_idx, data)
        data['data_root'] = self.data_root_path[dataset_idx]
        for transform in self.transforms:
            data = transform(data)

        return data
