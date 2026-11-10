# Copyright 2026 Limx Dynamics
"""LeRobot v3 options of ParquetDatasetV3 and ProcessParquetInputs."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from fluxvla.datasets.parquet_dataset_v3 import ParquetDatasetV3
from fluxvla.transforms.transform_inputs import ProcessParquetInputs

CAMERAS = ('cam_a', 'cam_b', 'wrist')
FPS = 15
# Episode -> (data file, global row range). File 2 is not referenced by
# metadata and restarts the global index, like the extra DROID files.
EPISODES = {0: (0, 0, 20), 1: (0, 20, 30), 2: (1, 30, 45)}
STATS = {'droid': {'unused': {}}}


def _write_parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _rows(episode, start, stop):
    rows = []
    for offset, index in enumerate(range(start, stop)):
        rows.append({
            'index': index,
            'episode_index': episode,
            'frame_index': offset,
            'timestamp': np.float32(offset / FPS),
            'task_index': episode,
            'action': [float(index), -float(index)],
            'observation.state': [float(index) + 0.5, 0.0],
            'language_instruction': f'first {episode}',
            'language_instruction_2': '',
        })
    return rows


def _make_dataset(root: Path, with_videos=False):
    meta = root / 'meta'
    meta.mkdir(parents=True)
    info = {
        'fps': FPS,
        'chunks_size': 1000,
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}'
        '.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/'
        'file-{file_index:03d}.mp4',
        'features': {},
    }
    (meta / 'info.json').write_text(json.dumps(info))
    (meta / 'stats.json').write_text('{}')
    _write_parquet(
        meta / 'tasks.parquet',
        pd.DataFrame({
            'task_index': [0, 1, 2],
            'task': ['task zero', 'task one', 'task two']
        }))
    episode_rows = []
    for episode, (file_index, start, stop) in EPISODES.items():
        row = {
            'episode_index': episode,
            'length': stop - start,
            'data/chunk_index': 0,
            'data/file_index': file_index,
            'dataset_from_index': start,
            'dataset_to_index': stop,
            'stats/action/mean': [0.0, 0.0],
        }
        for camera in CAMERAS:
            row[f'videos/{camera}/chunk_index'] = 0
            row[f'videos/{camera}/file_index'] = 0
            row[f'videos/{camera}/from_timestamp'] = start / FPS
            row[f'videos/{camera}/to_timestamp'] = stop / FPS
        episode_rows.append(row)
    _write_parquet(meta / 'episodes/chunk-000/file-000.parquet',
                   pd.DataFrame(episode_rows))
    by_file = {}
    for episode, (file_index, start, stop) in EPISODES.items():
        by_file.setdefault(file_index, []).extend(_rows(episode, start, stop))
    by_file[2] = _rows(9, 0, 5)
    for file_index, rows in by_file.items():
        _write_parquet(root / f'data/chunk-000/file-{file_index:03d}.parquet',
                       pd.DataFrame(rows))
    if with_videos:
        for camera_index, camera in enumerate(CAMERAS):
            _write_video(
                root / f'videos/{camera}/chunk-000/file-000.mp4',
                total_frames=45,
                base=camera_index * 60)
    return root


def _write_video(path, total_frames, base):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('mpeg4', rate=FPS)
        stream.width, stream.height = 32, 16
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = 4_000_000
        stream.codec_context.gop_size = 1
        for frame_number in range(total_frames):
            value = base + frame_number * 4
            image = np.full((16, 32, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _starts(tmp_path, values, name='starts.npy', dtype=np.uint32):
    path = tmp_path / name
    np.save(path, np.asarray(values, dtype=dtype))
    return str(path)


def _frozen(root, starts, **kwargs):
    options = dict(
        data_root_path=str(root),
        transforms=[],
        action_window_size=4,
        action_key='action',
        statistic_name='droid',
        window_start_idx=0,
        referenced_data_files_only=True,
        sample_indices_path=starts)
    options.update(kwargs)
    return ParquetDatasetV3(**options)


def test_referenced_files_skip_unreferenced_data(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    dataset = _frozen(root, _starts(tmp_path, [0, 25, 31]))
    assert len(dataset.dataset) == 45
    assert len(dataset) == 3
    assert dataset.episodes == []
    assert dataset.dataset[44]['index'] == 44


def test_frozen_windows_repeat_and_supervise_episode_end(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    dataset = _frozen(
        root, _starts(tmp_path, [0, 17, 27]), supervise_terminal_padding=True)
    first = dataset.__getitem__(0, STATS)
    np.testing.assert_array_equal(first['actions'][:, 0], [0, 1, 2, 3])
    assert first['actions'].dtype == np.float32
    end = dataset.__getitem__(1, STATS)
    np.testing.assert_array_equal(end['actions'][:, 0], [17, 18, 19, 19])
    np.testing.assert_array_equal(end['action_masks'], [1, 1, 1, 1])
    assert end['task_description'] == 'task zero'
    unsupervised = _frozen(root, _starts(tmp_path, [27], name='b.npy'))
    sample = unsupervised.__getitem__(0, STATS)
    np.testing.assert_array_equal(sample['actions'][:, 0], [27, 28, 29, 29])
    np.testing.assert_array_equal(sample['action_masks'], [1, 1, 1, 0])


def test_data_episodes_bind_global_indices_to_rows(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    dataset = _frozen(root, _starts(tmp_path, [30, 44]), data_episodes=[2])
    assert len(dataset.dataset) == 15
    sample = dataset.__getitem__(1, STATS)
    assert int(sample['index']) == 44
    np.testing.assert_array_equal(sample['actions'][:, 0], [44] * 4)
    with pytest.raises(ValueError, match='outside selected episodes'):
        _frozen(
            root, _starts(tmp_path, [5, 30], name='c.npy'), data_episodes=[2])


@pytest.mark.parametrize('values, message', [
    ([3, 3], 'strictly increasing'),
    ([], 'nonempty integer'),
    ([45], 'outside selected episodes'),
])
def test_invalid_frozen_starts_are_rejected(tmp_path, values, message):
    root = _make_dataset(tmp_path / 'droid')
    with pytest.raises(ValueError, match=message):
        _frozen(root, _starts(tmp_path, values))


def test_frozen_options_require_referenced_mode(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    starts = _starts(tmp_path, [0])
    with pytest.raises(ValueError, match='requires referenced'):
        _frozen(root, starts, referenced_data_files_only=False)
    with pytest.raises(ValueError, match='use_delta'):
        _frozen(root, starts, use_delta=True)
    with pytest.raises(ValueError, match='sample_indices_path'):
        _frozen(
            root,
            None,
            attach_episode_video_meta=True,
            referenced_data_files_only=True)


def _expected_choice(seed, epoch, episode):
    digest = hashlib.sha256(f'{seed}\0{epoch}\0{episode}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def test_episode_choices_follow_seed_epoch_and_episode(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    dataset = _frozen(
        root,
        _starts(tmp_path, [0, 5, 20]),
        episode_video_keys=[['cam_a', 'cam_b'], 'wrist'],
        episode_task_keys=['language_instruction', 'language_instruction_2'],
        selection_seed=7)
    for epoch in range(4):
        dataset.set_sample_epoch(epoch)
        for index, episode in [(0, 0), (1, 0), (2, 1)]:
            sample = dataset.__getitem__(index, STATS)
            choice = _expected_choice(7, epoch, episode)
            camera = ['cam_a', 'cam_b'][choice % 2]
            assert sample['lerobot_video_keys'] == [camera, 'wrist']
            task_key = ['language_instruction',
                        'language_instruction_2'][(choice // 2) % 2]
            assert sample['task_key'] == task_key
            expected_text = f'first {episode}' if task_key.endswith(
                'instruction') else ''
            assert sample['task_description'] == expected_text


def test_default_mode_keeps_legacy_window_and_metadata(tmp_path):
    root = _make_dataset(tmp_path / 'droid')
    (root / 'data/chunk-000/file-002.parquet').unlink()
    dataset = ParquetDatasetV3(
        data_root_path=str(root),
        transforms=[],
        action_window_size=4,
        action_key='action',
        statistic_name='droid',
        window_start_idx=0)
    assert len(dataset.episodes) == 3
    assert len(dataset) == 45
    sample = dataset.__getitem__(17, STATS)
    np.testing.assert_array_equal(sample['actions'][:, 0], [17, 18, 19, 19])
    np.testing.assert_array_equal(sample['action_masks'], [1, 1, 1, 0])
    assert sample['actions'].dtype == np.float32
    assert 'lerobot_episode_meta' not in sample
    assert sample['task_description'] == 'task zero'


def test_v3_videos_use_episode_offset_and_chosen_keys(tmp_path):
    root = _make_dataset(tmp_path / 'droid', with_videos=True)
    dataset = _frozen(
        root,
        _starts(tmp_path, [22]),
        attach_episode_video_meta=True,
        episode_video_keys=[['cam_a', 'cam_b'], 'wrist'],
        selection_seed=3,
        transforms=[
            dict(
                type='ProcessParquetInputs',
                parquet_keys=['observation.state', 'actions'],
                video_keys=['unused'],
                num_padding_imgs=1,
                video_backend='pyav')
        ])
    sample = dataset.__getitem__(0, STATS)
    chosen = ['cam_a', 'cam_b'][_expected_choice(3, 0, 1) % 2]
    # Episode 1 starts at merged frame 20; sample frame 2 is frame 22.
    bases = {'cam_a': 0, 'cam_b': 60, 'wrist': 120}
    images = sample['images']
    assert len(images) == 3
    assert images[0].shape == (3, 16, 32) and images[0].dtype == np.uint8
    for image, camera in zip(images[:2], [chosen, 'wrist']):
        assert abs(float(image.mean()) - (bases[camera] + 22 * 4)) < 3
    assert not images[2].any()
    np.testing.assert_array_equal(sample['img_masks'], [True, True, False])


def test_process_inputs_without_v3_meta_keeps_episode_chunk_paths(tmp_path):
    root = tmp_path / 'v2'
    _write_video(
        root / 'videos/chunk-000/cam_a/episode_000004.mp4',
        total_frames=6,
        base=10)
    data = {
        'info': {
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/'
            'episode_{episode_index:06d}.mp4',
            'chunks_size': 1000
        },
        'episode_index': 4,
        'timestamp': 3 / FPS,
        'data_root': str(root),
        'observation.state': [0.0],
    }
    result = ProcessParquetInputs(
        parquet_keys=['observation.state'],
        video_keys=['cam_a'],
        video_backend='pyav')(
            data)
    assert abs(float(result['images'][0].mean()) - (10 + 3 * 4)) < 3
