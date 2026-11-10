# Copyright 2026 Limx Dynamics
"""Opt-in strict training state and data-position restoration."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.utils.data import DataLoader

from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
from fluxvla.engines.runners import base_train_runner, fsdp_train_runner
from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner


def _full_checkpoint(**overrides):
    checkpoint = {
        'model': {},
        'global_step': 7,
        'epoch': 1,
        'optimizer_state_dict': {
            'state': {}
        },
        'scheduler_state_dict': {
            'last_epoch': 7
        },
    }
    checkpoint.update(overrides)
    return checkpoint


def _runner(strict, ema_decay=None):
    runner = object.__new__(FSDPTrainRunner)
    runner.resume_from = 'checkpoint.pt'
    runner.metric = SimpleNamespace(global_step=0)
    runner.current_epoch = 0
    runner.optimizer = mock.Mock()
    runner.lr_scheduler = mock.Mock()
    runner.ema_decay = ema_decay
    runner._ema_params = None
    runner.strict_training_state = strict
    runner._load_model_state = mock.Mock()
    runner._load_optimizer_state = mock.Mock(return_value=True)
    return runner


def _resume(runner, checkpoint):
    with mock.patch.object(
            base_train_runner.torch, 'load',
            return_value=checkpoint), mock.patch.object(
                base_train_runner.overwatch,
                'is_rank_zero',
                return_value=False), mock.patch.object(base_train_runner.dist,
                                                       'barrier') as barrier:
        runner.resume()
    return barrier


def test_strict_resume_restores_complete_state():
    runner = _runner(strict=True)
    _resume(runner, _full_checkpoint())
    assert runner.metric.global_step == 7
    assert runner.current_epoch == 1
    runner._load_optimizer_state.assert_called_once_with({'state': {}})
    runner.lr_scheduler.load_state_dict.assert_called_once_with(
        {'last_epoch': 7})


@pytest.mark.parametrize('missing', [
    'model', 'global_step', 'epoch', 'optimizer_state_dict',
    'scheduler_state_dict'
])
def test_strict_resume_rejects_missing_state(missing):
    checkpoint = _full_checkpoint()
    del checkpoint[missing]
    with pytest.raises(KeyError, match=missing):
        _resume(_runner(strict=True), checkpoint)
    # The default keeps the historical permissive behavior.
    _resume(_runner(strict=False), checkpoint)


@pytest.mark.parametrize('failure', [False, RuntimeError('bad optimizer')])
def test_strict_resume_propagates_optimizer_failure(failure):
    runner = _runner(strict=True)
    if isinstance(failure, Exception):
        runner._load_optimizer_state.side_effect = failure
    else:
        runner._load_optimizer_state.return_value = failure
    with pytest.raises(RuntimeError):
        _resume(runner, _full_checkpoint())
    assert runner._current_checkpoint_info is None

    permissive = _runner(strict=False)
    permissive._load_optimizer_state.side_effect = RuntimeError('bad')
    _resume(permissive, _full_checkpoint())
    assert permissive.metric.global_step == 7


def test_strict_resume_still_synchronizes_before_raising():
    runner = _runner(strict=True)
    runner._load_optimizer_state.side_effect = RuntimeError('bad optimizer')
    with mock.patch.object(
            base_train_runner.torch, 'load',
            return_value=_full_checkpoint()), mock.patch.object(
                base_train_runner.overwatch,
                'is_rank_zero',
                return_value=False), mock.patch.object(base_train_runner.dist,
                                                       'barrier') as barrier:
        with pytest.raises(RuntimeError, match='bad optimizer'):
            runner.resume()
    barrier.assert_called_once_with()


def test_strict_resume_propagates_scheduler_failure():
    runner = _runner(strict=True)
    runner.lr_scheduler.load_state_dict.side_effect = ValueError('bad lr')
    with pytest.raises(ValueError, match='bad lr'):
        _resume(runner, _full_checkpoint())
    permissive = _runner(strict=False)
    permissive.lr_scheduler.load_state_dict.side_effect = ValueError('bad')
    _resume(permissive, _full_checkpoint())


def test_strict_resume_checks_ema_state():
    with pytest.raises(KeyError, match='train_model'):
        _resume(_runner(strict=True, ema_decay=0.99), _full_checkpoint())
    with pytest.raises(ValueError, match='EMA decay'):
        _resume(
            _runner(strict=True, ema_decay=0.99),
            _full_checkpoint(train_model={}, ema_decay=0.999))
    with pytest.raises(ValueError, match='EMA is off'):
        _resume(_runner(strict=True), _full_checkpoint(train_model={}))

    runner = _runner(strict=True, ema_decay=0.99)
    runner._initialize_ema_parameters = mock.Mock()
    _resume(
        runner,
        _full_checkpoint(
            model={'ema': 1}, train_model={'raw': 1}, ema_decay=0.99))
    assert runner._load_model_state.call_args_list == [
        mock.call({'ema': 1}), mock.call({'raw': 1})
    ]
    runner._initialize_ema_parameters.assert_called_once_with()


def _step_runner(strict):
    runner = object.__new__(FSDPTrainRunner)
    runner.strict_training_state = strict
    runner.lr_scheduler = mock.Mock()
    runner.mixed_precision_dtype = torch.bfloat16
    runner.enable_mixed_precision_training = False
    runner.keep_params_fp32 = True
    runner.device_id = 0
    runner.grad_accumulation_steps = 1
    runner.metric = mock.Mock()
    runner.ema_decay = None
    weight = nn.Parameter(torch.ones(()))
    runner.vla = mock.Mock(return_value={'loss': weight * 2})
    runner._prepare_batch = mock.Mock(side_effect=lambda batch, *_: batch)
    runner.clip_grad_norm = mock.Mock()
    runner.optimizer = mock.Mock()
    runner.optimizer.step.side_effect = [
        RuntimeError('size mismatch in state'), None
    ]
    runner._reinit_optimizer = mock.Mock()
    return runner


def test_strict_training_step_never_rebuilds_optimizer():
    runner = _step_runner(strict=True)
    with mock.patch.object(
            base_train_runner.overwatch, 'is_rank_zero', return_value=False):
        with pytest.raises(RuntimeError, match='size mismatch'):
            runner._training_step({})
    runner._reinit_optimizer.assert_not_called()


def test_default_training_step_keeps_optimizer_rebuild_fallback():
    runner = _step_runner(strict=False)
    with mock.patch.object(
            base_train_runner.overwatch, 'is_rank_zero', return_value=False):
        runner._training_step({})
    runner._reinit_optimizer.assert_called_once_with()


def test_runner_forwards_strict_option():
    with mock.patch.object(
            base_train_runner.BaseTrainRunner, '__init__',
            return_value=None) as base_init, mock.patch.object(
                fsdp_train_runner.overwatch,
                'local_rank',
                return_value=0,
                create=True):
        FSDPTrainRunner(
            cfg={},
            max_grad_norm=1,
            collator={},
            sampler=None,
            metric={},
            strict_training_state=True)
        FSDPTrainRunner(
            cfg={}, max_grad_norm=1, collator={}, sampler=None, metric={})
    assert base_init.call_args_list[0].kwargs['strict_training_state']
    assert not base_init.call_args_list[1].kwargs['strict_training_state']


class _TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)
        self.output = nn.Linear(4, 2)


@pytest.fixture()
def single_process_group(tmp_path):
    if dist.is_initialized():
        assert dist.get_world_size() == 1
        yield
        return
    dist.init_process_group(
        'gloo',
        init_method='file://{}'.format(tmp_path / 'gloo-rendezvous'),
        rank=0,
        world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('strict', [False, True])
def test_fsdp_model_load_matches_keys_only_when_strict(single_process_group,
                                                       strict):
    runner = object.__new__(FSDPTrainRunner)
    runner.vla = FSDP(
        _TinyModel(),
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=torch.device('cpu'))
    runner.change_key_name = False
    runner.strict_training_state = strict
    runner.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
    runner.fsdp_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True)
    state = {
        key: torch.zeros_like(value)
        for key, value in _TinyModel().state_dict().items()
        if key != 'output.bias'
    }
    with mock.patch.object(
            fsdp_train_runner.overwatch, 'is_rank_zero', return_value=False):
        if strict:
            with pytest.raises(RuntimeError, match='output.bias'):
                runner._load_model_state(state)
        else:
            runner._load_model_state(state)
            with FSDP.summon_full_params(runner.vla):
                assert not runner.vla.block.weight.any()


class _EpochRecorder:

    def __init__(self):
        self.epochs = []

    def set_sample_epoch(self, epoch):
        self.epochs.append(epoch)

    def __getitem__(self, index, statistics):
        return index


class _PlainDataset:

    def __getitem__(self, index, statistics):
        return index


def test_wrapper_forwards_epoch_only_to_opted_in_datasets():
    wrapper = object.__new__(DistributedRepeatingDataset)
    recorder = _EpochRecorder()
    wrapper.is_grouped = False
    wrapper.is_list = True
    wrapper.datasets = [recorder, _PlainDataset()]
    wrapper._set_inner_sample_epoch(3)
    assert recorder.epochs == [3]

    grouped = object.__new__(DistributedRepeatingDataset)
    grouped.is_grouped = True
    grouped.is_list = False
    grouped.grouped_datasets = {'a': [_PlainDataset()], 'b': [recorder]}
    grouped._set_inner_sample_epoch(5)
    assert recorder.epochs == [3, 5]


def test_wrapper_iteration_sets_runner_epoch_before_sampling():
    recorder = _EpochRecorder()
    wrapper = object.__new__(DistributedRepeatingDataset)
    wrapper.is_grouped = False
    wrapper.is_list = False
    wrapper.dataset = recorder
    wrapper.dataset_statistics = {}
    wrapper.total_len = 4
    wrapper.shuffle = False
    wrapper.reshuffle_each_epoch = True
    wrapper.seed = 0
    wrapper.rank = 0
    wrapper.world_size = 1
    wrapper._epoch = 0
    wrapper._sharding_strategy = 'round_robin'
    wrapper._runner_epoch = torch.full((), 2, dtype=torch.int64)
    iterator = iter(wrapper)
    assert [next(iterator) for _ in range(4)] == [0, 1, 2, 3]
    assert recorder.epochs == [2]


class _IndexDataset:

    def __getitem__(self, index, statistics):
        return index


def _index_wrapper(total, strategy='round_robin', batch_size=None):
    wrapper = object.__new__(DistributedRepeatingDataset)
    wrapper.is_grouped = False
    wrapper.is_list = False
    wrapper.dataset = _IndexDataset()
    wrapper.dataset_statistics = {}
    wrapper.total_len = total
    wrapper.shuffle = True
    wrapper.reshuffle_each_epoch = True
    wrapper.seed = 5
    wrapper.rank = 0
    wrapper.world_size = 1
    wrapper._epoch = 0
    wrapper._sharding_strategy = strategy
    wrapper._per_rank_batch_size = batch_size
    for name, value in [('_runner_epoch', -1), ('_resume_epoch', -1),
                        ('_resume_rank_batches', 0),
                        ('_resume_batch_size', 0)]:
        setattr(wrapper, name,
                torch.full((), value, dtype=torch.int64).share_memory_())
    return wrapper


def _batches(wrapper, workers, batch_size, epoch, count):
    wrapper.set_epoch(epoch)
    # Lists, not tensors: worker tensors would share fds over a socket,
    # which the test network guard rejects. Batch order is unaffected.
    iterator = iter(
        DataLoader(
            wrapper,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=list))
    return [next(iterator) for _ in range(count)]


@pytest.mark.parametrize('consumed', [0, 4, 7, 40])
@pytest.mark.parametrize('strategy', ['round_robin', 'blockwise'])
@pytest.mark.parametrize('workers', [0, 1, 3])
def test_resume_position_continues_the_interrupted_stream(
        workers, strategy, consumed):
    batch_size = 4
    uninterrupted = _batches(
        _index_wrapper(37, strategy, batch_size), workers, batch_size, 2,
        consumed + 6)
    wrapper = _index_wrapper(37, strategy, batch_size)
    wrapper.set_resume_position(2, consumed, batch_size)
    resumed = _batches(wrapper, workers, batch_size, 2, 6)
    assert resumed == uninterrupted[consumed:]


def test_resume_position_only_affects_its_epoch():
    wrapper = _index_wrapper(20)
    wrapper.set_resume_position(1, 3, 4)
    assert _batches(wrapper, 2, 4, 2,
                    5) == _batches(_index_wrapper(20), 2, 4, 2, 5)
    with pytest.raises(ValueError, match='Resume position'):
        wrapper.set_resume_position(1, -1, 4)


class _ResumableDataset:
    seed = 42
    shuffle = True
    reshuffle_each_epoch = True

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 75

    def set_resume_position(self, *args):
        self.calls.append(args)


def _progress_runner(global_step=0, enabled=True):
    runner = object.__new__(FSDPTrainRunner)
    runner.resume_data_progress = enabled
    runner.per_device_batch_size = 4
    runner.grad_accumulation_steps = 2
    runner.per_device_num_workers = 3
    runner.dataset_sharding_strategy = 'round_robin'
    runner.steps_per_epoch = 10
    runner.metric = SimpleNamespace(global_step=global_step)
    runner.current_epoch = 0
    runner._epoch_step_count = 0
    runner._resume_epoch_step = 0
    runner._train_dataset = _ResumableDataset()
    return runner


@pytest.fixture()
def two_ranks():
    with mock.patch.object(
            base_train_runner.overwatch, 'world_size',
            return_value=2), mock.patch.object(
                base_train_runner.overwatch,
                'is_rank_zero',
                return_value=False):
        yield


@pytest.mark.parametrize('epoch, step, expected', [
    (3, 4, (3, 8, 4)),
    (3, 10, (4, 0, 4)),
])
def test_saved_data_progress_restores_position(two_ranks, epoch, step,
                                               expected):
    saving = _progress_runner()
    saving.current_epoch, saving._epoch_step_count = epoch, step
    state = saving._data_progress_state()
    assert state['world_size'] == 2 and state['dataset_seed'] == 42

    runner = _progress_runner(global_step=epoch * 10 + step)
    runner._pending_data_progress = state
    dataset = _ResumableDataset()
    runner._apply_data_progress(dataset)
    assert dataset.calls == [expected]
    assert runner.current_epoch == expected[0]
    assert runner._resume_epoch_step == expected[1] // 2


@pytest.mark.parametrize('field, value', [('num_workers', 1),
                                          ('world_size', 4),
                                          ('per_device_batch_size', 8),
                                          ('dataset_seed', 7)])
def test_data_progress_rejects_changed_settings(two_ranks, field, value):
    state = _progress_runner()._data_progress_state()
    state[field] = value
    runner = _progress_runner()
    runner._pending_data_progress = state
    with pytest.raises(ValueError, match=field):
        runner._apply_data_progress(_ResumableDataset())


def test_data_progress_rejects_inconsistent_steps(two_ranks):
    saving = _progress_runner()
    saving.current_epoch, saving._epoch_step_count = 1, 2
    runner = _progress_runner(global_step=99)
    runner._pending_data_progress = saving._data_progress_state()
    with pytest.raises(ValueError, match='global_step'):
        runner._apply_data_progress(_ResumableDataset())


def test_data_progress_requires_resumable_dataset(two_ranks):
    runner = _progress_runner()
    runner._pending_data_progress = runner._data_progress_state()

    class Plain(_ResumableDataset):
        set_resume_position = None

    with pytest.raises(TypeError, match='cannot resume'):
        runner._apply_data_progress(Plain())


def test_data_progress_is_ignored_when_disabled(two_ranks):
    runner = _progress_runner(enabled=False)
    runner._pending_data_progress = runner._data_progress_state()
    dataset = _ResumableDataset()
    runner._apply_data_progress(dataset)
    assert dataset.calls == []


def test_resume_keeps_data_progress_and_strict_requires_it():
    progress = {'version': 1, 'epoch': 0, 'epoch_step': 7}
    runner = _runner(strict=True)
    runner.resume_data_progress = True
    _resume(runner, _full_checkpoint(data_progress=progress))
    assert runner._pending_data_progress == progress
    strict = _runner(strict=True)
    strict.resume_data_progress = True
    with pytest.raises(KeyError, match='data_progress'):
        _resume(strict, _full_checkpoint())


def test_step_loop_continues_epoch_after_restored_position():

    class Metric:
        global_step = 8

        def commit(self, **values):
            self.global_step = values.get('global_step', self.global_step)

        def push(self):
            return ''

        def get_status(self):
            return ''

    runner = object.__new__(FSDPTrainRunner)
    runner.metric = Metric()
    runner.max_steps = 12
    runner.steps_per_epoch = 5
    runner.current_epoch = 1
    runner._resume_epoch_step = 3
    runner._loss_accumulator = []
    runner.evaluator = None
    iterators = []

    def step(loader, iterator, sampler):
        iterators.append(iterator)
        return 0.0, 'live'

    runner._run_accumulated_training_step = step
    runner._get_log_lr = lambda: 0.0
    runner._should_save_step_checkpoint = lambda: False
    runner._get_checkpoint_path = lambda: 'checkpoint'
    with mock.patch.object(
            base_train_runner.overwatch, 'is_rank_zero', return_value=False):
        runner._run_step_based(None, None, None)
    # Epoch 1 already had 3 steps, so it ends after 2 more.
    assert iterators == [None, 'live', None, 'live']
    assert runner.current_epoch == 2 and runner._epoch_step_count == 2


def test_global_batch_must_match_the_recipe():
    runner = object.__new__(FSDPTrainRunner)
    runner.per_device_batch_size = 32
    runner.grad_accumulation_steps = 1
    runner.global_batch_size = 128
    with mock.patch.object(
            base_train_runner.overwatch, 'world_size', return_value=4):
        with pytest.raises(ValueError, match='grad_accumulation_steps'):
            runner._check_global_batch_size(256)
        runner._check_global_batch_size(None)
        runner.grad_accumulation_steps = 2
        runner.global_batch_size = 256
        runner._check_global_batch_size(256)


@pytest.mark.parametrize('interval', [None, 10])
def test_cleanup_keeps_only_milestone_weights(tmp_path, interval):
    runner = object.__new__(FSDPTrainRunner)
    runner.max_keep_ckpts = 1
    runner.milestone_interval = interval
    for order, step in enumerate((5, 10, 15)):
        for suffix in ('.pt', '.safetensors'):
            path = tmp_path / f'step-{step:06d}-epoch-00-loss=1.0{suffix}'
            path.write_bytes(b'x')
            os.utime(path, (order, order))
    runner._cleanup_old_checkpoints(str(tmp_path))
    expected = {
        'step-000015-epoch-00-loss=1.0.pt',
        'step-000015-epoch-00-loss=1.0.safetensors'
    }
    if interval:
        expected.add('step-000010-epoch-00-loss=1.0.safetensors')
    assert {path.name for path in tmp_path.iterdir()} == expected


@pytest.mark.parametrize('enabled', [False, True])
def test_checkpoint_records_data_progress_only_when_enabled(
        single_process_group, tmp_path, enabled):
    runner = object.__new__(FSDPTrainRunner)
    runner.vla = FSDP(
        _TinyModel(),
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=torch.device('cpu'))
    runner.optimizer = torch.optim.AdamW(runner.vla.parameters(), lr=1e-3)
    runner.lr_scheduler = None
    runner.tokenizer = None
    runner.change_key_name = False
    runner.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
    runner.fsdp_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True)
    runner.max_keep_ckpts = 2
    runner.resume_data_progress = enabled
    runner._data_progress_state = lambda: {'version': 1, 'epoch_step': 7}
    with mock.patch.object(
            fsdp_train_runner.overwatch,
            'is_rank_zero', return_value=True), mock.patch.object(
                torch.cuda,
                'synchronize'), mock.patch.object(torch.cuda, 'empty_cache'):
        runner.save_checkpoint(
            tmp_path, global_step=27, epoch=2, train_loss=1.0)
    checkpoint = torch.load(
        next((tmp_path / 'checkpoints').glob('step-*.pt')),
        map_location='cpu',
        weights_only=True)
    assert ('data_progress' in checkpoint) is enabled
    if enabled:
        assert checkpoint['data_progress']['epoch_step'] == 7
