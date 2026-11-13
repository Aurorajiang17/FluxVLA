"""Fixed model ownership for standard, continuously sampled requests."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

from .continuous_sampling import MAX_AUDIT_BYTES, standard_request_context


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


@dataclass(frozen=True)
class PredictionTicket:
    rank: int
    global_sequence: int
    stream_sequence: int
    request_id: str
    episode_id: str
    sequence: int


@dataclass(frozen=True)
class PreparedReport:
    event_type: str
    run_session_id: str
    payload: dict
    jobs: dict | None = None


class FixedWorkerContinuousSession:
    """Own one bounded audit and a permanent model/client pair per rank."""

    def __init__(self,
                 expected_workers,
                 audit_path,
                 max_audit_bytes,
                 deployment_id,
                 process_seed,
                 config_sha256=''):
        self.expected_workers = _integer(expected_workers, 'expected_workers',
                                         1)
        self._limit = _integer(max_audit_bytes, 'max_audit_bytes', 1)
        if self._limit > MAX_AUDIT_BYTES:
            raise ValueError('max_audit_bytes exceeds 128 MiB')
        _string(deployment_id, 'deployment_id')
        self._process_seed = _integer(process_seed, 'process_seed', 0)
        if self._process_seed >= 2**32:
            raise ValueError('process_seed must be < 2**32')
        self._bytes = 0
        self._failed = False
        self._ended = False
        self._run_id = None
        self._jobs = {}
        self._next = {}
        self._models = {}
        self._routes = {}
        self._owners = {}
        self._tickets = {}
        self._active = {}
        self._completed = set()
        self._started = set()
        self._finished = set()
        self._accepted_by_episode = {}
        self._stream_sequences = {}
        self._global_sequence = 0
        self._file = open(audit_path, 'xb')
        try:
            self._write(
                dict(
                    event='started',
                    schema_version=2,
                    routing='fixed_worker',
                    deployment_id=deployment_id,
                    expected_workers=self.expected_workers,
                    process_seed=process_seed,
                    config_sha256=config_sha256))
        except BaseException:
            self._file.close()
            raise

    def _write(self, record):
        data = (json.dumps(record, sort_keys=True) + '\n').encode()
        if self._bytes + len(data) > self._limit:
            raise RuntimeError('Continuous sampling audit capacity exceeded')
        if self._file.write(data) != len(data):
            raise OSError('Short continuous sampling audit write')
        self._file.flush()
        self._bytes += len(data)

    @contextmanager
    def _operation(self):
        try:
            if self._failed or self._file.closed:
                raise RuntimeError('Continuous sampling session is sealed')
            yield
        except BaseException as exc:
            self.fail(str(exc))
            raise

    def bind_model(self, rank, worker_id, instance_id, device):
        with self._operation():
            rank = _integer(rank, 'rank', 0)
            if rank >= self.expected_workers:
                raise ValueError('rank is outside the configured model pool')
            for value, name in [(worker_id, 'worker_id'),
                                (instance_id, 'process_instance_id'),
                                (device, 'device')]:
                _string(value, name)
            if rank in self._models or any(
                    worker_id == m[0] or instance_id == m[1]
                    for m in self._models.values()):
                raise RuntimeError(
                    'Continuous model instance cannot be replaced')
            self._write(
                dict(
                    event='model_ready',
                    rank=rank,
                    worker_id=worker_id,
                    process_instance_id=instance_id,
                    device=device,
                    process_seed=self._process_seed))
            self._models[rank] = (worker_id, instance_id, device)
            self._stream_sequences[rank] = 0

    def _build_jobs(self, run_id, payload):
        if self._run_id is not None:
            raise ValueError('Only one evaluation run is supported')
        if payload.get('strict_worker_assignment') is not True:
            raise ValueError(
                'Fixed continuous service requires fixed scheduling')
        workers = _integer(
            payload.get('parallel_workers'), 'parallel_workers', 1)
        if workers != self.expected_workers:
            raise ValueError(
                'parallel_workers must match the model worker count')
        if payload.get('resumed_from') or payload.get('restored_episodes'):
            raise ValueError('Continuous service does not support recovery')
        tasks = payload.get('tasks')
        if not isinstance(tasks, list) or not tasks:
            raise ValueError('tasks must be a nonempty ordered list')
        ids = []
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError('Each task must be a mapping')
            ids.append(_string(task.get('task_id'), 'task_id'))
        if len(set(ids)) != len(ids) or payload.get('task_ids') != ids:
            raise ValueError('Task manifest must have unique ordered task IDs')
        if _integer(payload.get('total_tasks'), 'total_tasks', 1) != len(ids):
            raise ValueError('total_tasks disagrees with the task manifest')
        counts = payload.get('episodes_per_task_by_task')
        if not isinstance(counts, Mapping) or set(counts) != set(ids):
            raise ValueError(
                'Episode quotas must exactly match the task manifest')
        counts = {
            task: _integer(counts[task], 'episode quota', 1)
            for task in ids
        }
        if _integer(payload.get('total_episodes'), 'total_episodes', 1) != sum(
                counts.values()):
            raise ValueError(
                'total_episodes disagrees with the episode quotas')
        seed = _integer(payload.get('base_seed'), 'base_seed')
        jobs = {}
        for task in ids:
            for episode_index in range(counts[task]):
                ordinal = len(jobs) + 1
                episode_id = f'{run_id}:{ordinal}'
                jobs[episode_id] = dict(
                    episode_id=episode_id,
                    global_episode_number=ordinal,
                    task_id=task,
                    episode_index=episode_index,
                    policy_seed=seed + ordinal - 1,
                    rank=(ordinal - 1) % self.expected_workers)
        return jobs

    def _check_episode_report(self, payload):
        episode_id = payload.get('episode_id')
        if episode_id not in self._jobs:
            raise ValueError('Unknown evaluation episode')
        job = self._jobs[episode_id]
        for field in ('task_id', 'episode_index', 'policy_seed'):
            value = payload.get(field)
            if field != 'task_id':
                value = _integer(value, field)
            if value != job[field]:
                raise ValueError(f'Episode report changed {field}')
        if episode_id in self._finished:
            raise ValueError('Episode report reused a completed episode')
        queue = self._next[job['rank']]
        if not queue or queue[0] != episode_id:
            raise ValueError('Episode report changed fixed worker order')
        return episode_id

    def prepare_report(self, event_type, run_session_id, payload):
        """Validate without advancing the report or prediction state."""
        with self._operation():
            run_id = _string(run_session_id, 'run_session_id')
            if not isinstance(payload, Mapping):
                raise ValueError('Report payload must be a mapping')
            body = dict(payload)
            if event_type == 'run_start':
                jobs = self._build_jobs(run_id, body)
                return PreparedReport(event_type, run_id, body, jobs)
            if run_id != self._run_id or self._ended:
                raise ValueError(
                    'Report belongs to an inactive evaluation run')
            if event_type in {'episode_start', 'episode_end'}:
                episode_id = self._check_episode_report(body)
                if event_type == 'episode_start':
                    if episode_id in self._started:
                        raise ValueError('Duplicate episode_start')
                else:
                    if episode_id not in self._started:
                        raise ValueError('episode_end requires episode_start')
                    tickets = self._accepted_by_episode.get(episode_id, [])
                    if any(t.request_id not in self._completed
                           for t in tickets):
                        raise ValueError('Episode has unfinished predictions')
                    metadata = body.get('prediction_metadata')
                    if not isinstance(metadata, list):
                        raise ValueError('prediction_metadata must be a list')
                    actual = []
                    for item in metadata:
                        if not isinstance(item, Mapping):
                            raise ValueError(
                                'Prediction metadata must be a mapping')
                        actual.append(
                            (item.get('request_id'), item.get('episode_id'),
                             _integer(item.get('sequence'), 'sequence', 1)))
                    expected = [(t.request_id, t.episode_id, t.sequence)
                                for t in tickets]
                    if actual != expected or _integer(
                            body.get('model_calls'), 'model_calls',
                            0) != len(tickets):
                        raise ValueError(
                            'Episode prediction metadata disagrees with audit')
            elif event_type == 'run_end':
                total = len(self._jobs)
                if _integer(body.get('total_episodes'), 'total_episodes',
                            1) != total:
                    raise ValueError('run_end changed total_episodes')
                if _integer(
                        body.get('completed_episodes'), 'completed_episodes',
                        0) != len(self._finished):
                    raise ValueError('run_end changed completed_episodes')
                if body.get('status') == 'finished':
                    if (len(self._finished) != total or self._active
                            or len(self._completed) != len(self._tickets)):
                        raise ValueError('Finished run has unfinished work')
                elif body.get('status') not in {
                        'failed', 'cancelled', 'interrupted'
                }:
                    raise ValueError('Unsupported run_end status')
            else:
                raise ValueError('Unknown evaluation report event')
            return PreparedReport(event_type, run_id, body)

    def commit_report(self, prepared_report):
        """Commit only after the authoritative report writer accepts it."""
        with self._operation():
            kind, body = prepared_report.event_type, prepared_report.payload
            if kind == 'run_start':
                self._write(
                    dict(
                        event='run_started',
                        run_session_id=prepared_report.run_session_id,
                        total_episodes=len(prepared_report.jobs)))
                self._run_id = prepared_report.run_session_id
                self._jobs = prepared_report.jobs
                self._next = {
                    rank: deque()
                    for rank in range(self.expected_workers)
                }
                for episode_id, job in self._jobs.items():
                    self._next[job['rank']].append(episode_id)
            elif kind == 'episode_start':
                self._started.add(body['episode_id'])
            elif kind == 'episode_end':
                episode_id = body['episode_id']
                rank = self._jobs[episode_id]['rank']
                self._write(
                    dict(
                        event='episode_finished',
                        rank=rank,
                        episode_id=episode_id))
                self._next[rank].popleft()
                self._finished.add(episode_id)
            elif kind == 'run_end':
                if body['status'] != 'finished':
                    self.fail(f"Evaluation ended with status {body['status']}")
                else:
                    self._write(
                        dict(
                            event='run_finished',
                            run_session_id=self._run_id,
                            completed_episodes=len(self._finished)))
                    self._ended = True

    def accept(self, route, header, payload):
        with self._operation():
            context = standard_request_context(header)
            if len(self._models) != self.expected_workers:
                raise RuntimeError('Continuous models are not all ready')
            if self._run_id is None or self._ended:
                raise RuntimeError('Continuous evaluation run is not active')
            episode_id = context['episode_id']
            if episode_id not in self._jobs:
                raise ValueError('Unknown evaluation episode')
            job = self._jobs[episode_id]
            rank = job['rank']
            if not self._next[rank] or self._next[rank][0] != episode_id:
                raise ValueError(
                    'Prediction changed fixed worker episode order')
            if header['policy_seed'] != job['policy_seed']:
                raise ValueError('Prediction changed the planned policy_seed')
            if (route in self._routes and self._routes[route] != rank
                    or rank in self._owners and self._owners[rank] != route):
                raise RuntimeError(
                    'Continuous rank is owned by another client')
            if rank in self._active:
                raise RuntimeError(
                    'Continuous rank accepts one request at a time')
            request_id = header['request_id']
            if request_id in self._tickets:
                raise ValueError('Duplicate completed or active request_id')
            prior = self._accepted_by_episode.get(episode_id, [])
            if context['sequence'] != len(prior) + 1:
                raise ValueError('sequence must advance by one')
            ticket = PredictionTicket(rank, self._global_sequence + 1,
                                      self._stream_sequences[rank] + 1,
                                      request_id, episode_id,
                                      context['sequence'])
            worker_id, instance_id, _ = self._models[rank]
            route_hash = hashlib.sha256(b'\0'.join(route)).hexdigest()
            self._write(
                dict(
                    event='accepted',
                    **ticket.__dict__,
                    worker_id=worker_id,
                    process_instance_id=instance_id,
                    route_sha256=route_hash,
                    policy_seed=int(header['policy_seed']),
                    payload_sha256=hashlib.sha256(payload).hexdigest()))
            self._global_sequence = ticket.global_sequence
            self._stream_sequences[rank] = ticket.stream_sequence
            self._routes[route] = rank
            self._owners[rank] = route
            self._tickets[request_id] = ticket
            self._active[rank] = ticket
            self._accepted_by_episode.setdefault(episode_id, []).append(ticket)
            return ticket

    def complete(self, request_id, worker_id, instance_id, process_sequence):
        with self._operation():
            ticket = self._tickets.get(request_id)
            if ticket is None or self._active.get(ticket.rank) != ticket:
                raise ValueError('Unexpected completed request_id')
            model = self._models[ticket.rank]
            if (worker_id, instance_id) != model[:2]:
                raise ValueError(
                    'Prediction returned a different model instance')
            if _integer(process_sequence, 'process_sequence',
                        1) != ticket.stream_sequence:
                raise ValueError(
                    'Model process sequence disagrees with accepted request')
            self._write(
                dict(
                    event='completed',
                    **ticket.__dict__,
                    worker_id=worker_id,
                    process_instance_id=instance_id,
                    process_sequence=int(process_sequence)))
            self._completed.add(request_id)
            del self._active[ticket.rank]

    def check_model_identity(self, rank, worker_id, instance_id):
        with self._operation():
            if self._models.get(rank, ())[:2] != (worker_id, instance_id):
                raise ValueError('Heartbeat changed the model instance')

    def fail(self, reason):
        if self._failed:
            return
        self._failed = True
        try:
            self._write(
                dict(
                    event='failed',
                    reason=reason,
                    active_requests=[
                        t.__dict__ for t in self._active.values()
                    ]))
        except Exception:
            print(
                f'[FluxVLA] continuous audit failed: {reason}',
                file=sys.stderr)

    def close(self):
        if self._file.closed:
            return
        try:
            if not self._failed:
                if not self._ended or self._active:
                    self.fail('Stopped before a complete evaluation run_end')
                else:
                    try:
                        self._write(
                            dict(
                                event='closed',
                                requests=self._global_sequence,
                                requests_by_rank=self._stream_sequences))
                    except BaseException as exc:
                        self.fail(str(exc))
                        raise
        finally:
            self._file.close()
