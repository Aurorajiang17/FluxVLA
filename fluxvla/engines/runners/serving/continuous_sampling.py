"""Explicit ownership and bounded auditing for continuous model sampling."""
from __future__ import annotations
import hashlib
import json
import os
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

MAX_AUDIT_BYTES = 128 * 1024 * 1024
RNG_FIELDS = frozenset({
    'sampling_mode', 'rng_state', 'rng_format', 'process_instance_id',
    'process_sequence', 'process_seed', 'accept_standard_process'
})


def standard_request_context(header):
    forbidden = RNG_FIELDS.intersection(header)
    if forbidden:
        raise ValueError('Continuous service rejects caller RNG fields: '
                         f'{sorted(forbidden)}')
    for key in ('request_id', 'episode_id'):
        if not isinstance(header.get(key), str) or not header[key]:
            raise ValueError(f'{key} must be a nonempty string')
    for key in ('sequence', 'policy_seed'):
        if isinstance(header.get(key),
                      bool) or not isinstance(header.get(key), Integral):
            raise ValueError(f'{key} must be an integer')
    if header['sequence'] < 1:
        raise ValueError('sequence must be positive')
    return dict(
        episode_id=header['episode_id'], sequence=int(header['sequence']))


def continuous_sampling_config(server_cfg, expected_workers=1):
    """Validate settings without opening the broker-owned audit file."""
    value = server_cfg.get('continuous_sampling', {})
    if not isinstance(value, Mapping):
        raise TypeError('continuous_sampling must be a mapping')
    enabled = value.get('enabled', False)
    if not isinstance(enabled, bool):
        raise TypeError('continuous_sampling.enabled must be a bool')
    if not enabled:
        return None
    unknown = set(value) - {
        'enabled', 'routing', 'audit_path', 'max_audit_bytes'
    }
    if unknown:
        raise ValueError(
            f'Unknown continuous_sampling keys: {sorted(unknown)}')
    runner = server_cfg.get('runner', {})
    if runner.get('sampling_mode') != 'process_noise_stream_v1':
        raise ValueError('Continuous service requires process_noise_stream_v1')
    seed = runner.get('process_seed')
    if isinstance(
            seed,
            bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError(
            'Continuous service requires a supported process_seed')
    routing = value.get('routing', 'single_client')
    if routing not in {'single_client', 'fixed_worker'}:
        raise ValueError('Unsupported continuous_sampling.routing')
    if isinstance(expected_workers, bool) or not isinstance(
            expected_workers, Integral) or expected_workers < 1:
        raise ValueError('expected_workers must be a positive integer')
    if routing == 'single_client' and expected_workers != 1:
        raise ValueError('Continuous service requires one model worker')
    if routing == 'fixed_worker' and server_cfg.get(
            'evaluation_reporting', {}).get('enabled') is not True:
        raise ValueError(
            'Fixed continuous service requires evaluation_reporting')
    if runner.get('process_resume') or runner.get('process_checkpoint_dir'):
        raise ValueError('Continuous service does not support recovery')
    path = value.get('audit_path')
    if not isinstance(path, (str, os.PathLike)) or not str(path):
        raise ValueError('continuous_sampling.audit_path is required')
    path = Path(path).expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f'Audit parent does not exist: {path.parent}')
    limit = value.get('max_audit_bytes', MAX_AUDIT_BYTES)
    if isinstance(limit, bool) or not isinstance(
            limit, Integral) or not 0 < limit <= MAX_AUDIT_BYTES:
        raise ValueError('max_audit_bytes must be in (0, 128 MiB]')
    canonical = json.dumps(
        dict(server_cfg), sort_keys=True, default=str).encode()
    result = dict(
        audit_path=path,
        max_audit_bytes=int(limit),
        process_seed=int(seed),
        config_sha256=hashlib.sha256(canonical).hexdigest())
    if routing == 'fixed_worker':
        result.update(routing=routing)
    return result


class ContinuousSamplingSession:
    """Seal after any uncertain request; only the broker owns the file."""

    def __init__(self,
                 audit_path,
                 max_audit_bytes,
                 deployment_id,
                 process_seed,
                 config_sha256=''):
        self._file = open(audit_path, 'xb')
        self._limit, self._bytes = max_audit_bytes, 0
        self._failed, self._route, self._model = False, None, None
        self._seen_requests, self._seen_episodes = set(), set()
        self._episode, self._sequence, self._global_sequence = None, 0, 0
        self._active = None
        try:
            self._write(
                dict(
                    event='started',
                    deployment_id=deployment_id,
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

    def _require_open(self):
        if self._failed or self._file.closed:
            raise RuntimeError('Continuous sampling session is sealed')

    def bind_model(self, worker_id, instance_id):
        try:
            self._require_open()
            if self._model is not None:
                raise RuntimeError(
                    'Continuous model instance cannot be replaced')
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError('process_instance_id is required')
            self._write(
                dict(
                    event='model_ready',
                    worker_id=worker_id,
                    process_instance_id=instance_id))
            self._model = (worker_id, instance_id)
        except BaseException as exc:
            self.fail(str(exc))
            raise

    def accept(self, route, header, payload):
        try:
            self._require_open()
            context = standard_request_context(header)
            if self._model is None:
                raise RuntimeError('Continuous model is not ready')
            if self._route is not None and route != self._route:
                raise RuntimeError(
                    'Continuous endpoint is owned by another client')
            if self._active is not None:
                raise RuntimeError(
                    'Continuous endpoint accepts one request at a time')
            request = header['request_id']
            episode, sequence = context['episode_id'], context['sequence']
            if request in self._seen_requests:
                raise ValueError('Duplicate completed or active request_id')
            if episode == self._episode:
                if sequence != self._sequence + 1:
                    raise ValueError('sequence must advance by one')
            elif sequence != 1 or episode in self._seen_episodes:
                raise ValueError(
                    'New episode must start at sequence one and not be reused')
            next_global = self._global_sequence + 1
            self._write(
                dict(
                    event='accepted',
                    global_sequence=next_global,
                    request_id=request,
                    episode_id=episode,
                    sequence=sequence,
                    policy_seed=int(header['policy_seed']),
                    payload_sha256=hashlib.sha256(payload).hexdigest()))
            self._route = route
            self._seen_requests.add(request)
            self._seen_episodes.add(episode)
            self._episode, self._sequence = episode, sequence
            self._global_sequence, self._active = next_global, request
            return next_global
        except BaseException as exc:
            self.fail(str(exc))
            raise

    def complete(self, request_id):
        try:
            self._require_open()
            if self._active != request_id:
                raise RuntimeError('Unexpected completed request_id')
            self._write(
                dict(
                    event='completed',
                    request_id=request_id,
                    global_sequence=self._global_sequence))
            self._active = None
        except BaseException as exc:
            self.fail(str(exc))
            raise

    def fail(self, reason):
        if self._failed:
            return
        self._failed = True
        try:
            self._write(
                dict(event='failed', reason=reason, request_id=self._active))
        except Exception:
            import sys
            print(
                f'[FluxVLA] continuous audit failed: {reason}',
                file=sys.stderr)

    def close(self):
        if self._file.closed:
            return
        try:
            if not self._failed:
                if self._active is not None:
                    self.fail('Stopped with an unfinished prediction')
                else:
                    try:
                        self._write(
                            dict(
                                event='closed',
                                requests=self._global_sequence))
                    except BaseException as exc:
                        self.fail(str(exc))
                        raise
        finally:
            self._file.close()
