"""Stateless, multi-node ZMQ inference service for FluxThemis."""

from __future__ import annotations
import copy
import hashlib
import multiprocessing
import os
import signal
import socket as network_socket
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_sampling import (ContinuousSamplingSession,
                                  continuous_sampling_config,
                                  standard_request_context)
from .fixed_worker_sampling import FixedWorkerContinuousSession
from .serving_runner import (build_serving_runner_from_config, config_get,
                             get_server_config, require_mapping,
                             resolve_checkpoint_path,
                             resolve_report_config_path,
                             resolve_report_result_root)
from .zmq_protocol import (decode_frame, decode_header, empty_payload,
                           encode_frame, encode_header)


@dataclass
class _PendingPrediction:
    route: tuple[bytes, ...]
    request_id: str
    policy_seed: int
    unnorm_key: str
    payload: bytes
    deadline: float
    sampling_context: dict[str, Any] = field(default_factory=dict)
    request_context: dict[str, Any] = field(default_factory=dict)
    ticket: Any = None


@dataclass
class _ModelWorker:
    identity: bytes
    worker_id: str
    node_id: str
    device: str
    is_ready: bool = False
    prediction: _PendingPrediction | None = None
    rank: int | None = None
    last_seen: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.is_ready and self.prediction is None


class FluxVLAInferenceServer:
    """Route predictions to model workers and events to the report writer."""

    def __init__(self,
                 frontend_bind: str,
                 backend_bind: str,
                 evaluation_report_writer: Any = None,
                 expected_workers: int = 1,
                 max_pending_requests: int = 128,
                 request_timeout_s: float = 120.0,
                 startup_timeout_s: float = 900.0,
                 deployment_id: str | None = None,
                 continuous_sampling: Mapping[str, Any] | None = None) -> None:
        self.frontend_bind = _endpoint(frontend_bind, 'frontend_bind')
        self.backend_bind = _endpoint(backend_bind, 'backend_bind')
        self.evaluation_report_writer = evaluation_report_writer
        self.expected_workers = _positive_int(expected_workers,
                                              'expected_workers')
        self.max_pending_requests = _positive_int(max_pending_requests,
                                                  'max_pending_requests')
        self.request_timeout_s = _positive_float(request_timeout_s,
                                                 'request_timeout_s')
        self.startup_timeout_s = _positive_float(startup_timeout_s,
                                                 'startup_timeout_s')
        self.deployment_id = deployment_id or uuid.uuid4().hex
        options = dict(continuous_sampling or {})
        routing = options.pop('routing', 'single_client')
        if routing not in {'single_client', 'fixed_worker'}:
            raise ValueError('Unsupported continuous_sampling.routing')
        self._fixed = (
            continuous_sampling is not None and routing == 'fixed_worker')
        if self._fixed and evaluation_report_writer is None:
            raise ValueError(
                'Fixed continuous service requires evaluation_reporting')
        if (continuous_sampling is not None and not self._fixed
                and self.expected_workers != 1):
            raise ValueError('Continuous service requires one model worker')
        if self._fixed:
            self._continuous = FixedWorkerContinuousSession(
                expected_workers=self.expected_workers,
                deployment_id=self.deployment_id,
                **options)
        else:
            self._continuous = (None if continuous_sampling is None else
                                ContinuousSamplingSession(
                                    deployment_id=self.deployment_id,
                                    **options))

        self._workers: dict[bytes, _ModelWorker] = {}
        self._worker_ids: set[str] = set()
        self._idle_workers: deque[bytes] = deque()
        self._pending_predictions: deque[_PendingPrediction] = deque()
        self._predictions_by_id: dict[str, _PendingPrediction] = {}
        self._shutdown_started = False
        self._frontend = None
        self._backend = None
        self._context = None

    @property
    def ready_workers(self) -> int:
        return sum(worker.is_ready for worker in self._workers.values())

    @property
    def status(self) -> str:
        if self._shutdown_started:
            return 'stopping'
        ready = self.ready_workers == self.expected_workers
        return 'ready' if ready else 'starting'

    def run(self) -> None:
        import zmq

        try:
            self._context = zmq.Context()
            self._frontend = self._context.socket(zmq.ROUTER)
            self._backend = self._context.socket(zmq.ROUTER)
            for current in (self._frontend, self._backend):
                current.setsockopt(zmq.LINGER, 0)
                current.setsockopt(zmq.SNDHWM, self.max_pending_requests * 2)
                current.setsockopt(zmq.RCVHWM, self.max_pending_requests * 2)
            self._frontend.bind(self.frontend_bind)
            self._backend.bind(self.backend_bind)
            poller = zmq.Poller()
            poller.register(self._frontend, zmq.POLLIN)
            poller.register(self._backend, zmq.POLLIN)
            self._install_signal_handlers()
            startup_deadline = time.monotonic() + self.startup_timeout_s
            print(
                '[FluxVLA] inference server listening '
                f'frontend={self.frontend_bind} backend={self.backend_bind}',
                flush=True)
            while True:
                events = dict(poller.poll(timeout=200))
                if self._frontend in events:
                    self._receive_frontend()
                if self._backend in events:
                    self._receive_backend()
                self._expire_predictions()
                self._check_idle_workers(time.monotonic())
                self._dispatch_predictions()
                if (self.status == 'starting'
                        and time.monotonic() >= startup_deadline):
                    raise TimeoutError(
                        f'only {self.ready_workers}/'
                        f'{self.expected_workers} model workers became '
                        'ready')
                if self._shutdown_started and not self._predictions_by_id:
                    break
        except BaseException as exc:
            if self._continuous is not None:
                self._continuous.fail(str(exc))
            if self._fixed:
                for pending in self._predictions_by_id.values():
                    try:
                        self._send_frontend_error(pending.route,
                                                  pending.request_id, exc)
                    except Exception:
                        pass
            raise
        finally:
            try:
                self._begin_shutdown()
                if self._continuous is not None:
                    self._continuous.close()
            finally:
                for current in (self._frontend, self._backend):
                    if current is not None:
                        current.close(linger=1000)
                if self._context is not None:
                    self._context.term()

    def _receive_frontend(self) -> None:
        frames = self._frontend.recv_multipart()
        if len(frames) < 4 or frames[-3] != b'':
            return
        route = tuple(frames[:-2])
        request_id = ''
        message_type = None
        try:
            header = decode_header(frames[-2])
            if isinstance(header.get('request_id'), str):
                request_id = header['request_id']
            payload = frames[-1]
            message_type = header['type']
            if message_type == 'status':
                self._reply_status(route, header)
            elif message_type == 'predict_action':
                self._validate_deployment(header)
                self._queue_prediction(route, header, payload)
            elif message_type == 'report_event':
                self._validate_deployment(header)
                self._handle_report(route, header, payload)
            else:
                raise ValueError(f'Unknown endpoint: {message_type}')
        except Exception as exc:
            self._send_frontend_error(route, request_id=request_id, error=exc)
            if (self._continuous is not None
                    and (message_type == 'predict_action'
                         or self._fixed and message_type != 'status')):
                self._continuous.fail(str(exc))
                raise

    def _reply_status(self, route: tuple[bytes, ...],
                      request: Mapping[str, Any]) -> None:
        payload = {
            'status': self.status,
            'deployment_id': self.deployment_id,
            'server_mode': 'stateless',
            'evaluation_reporting': self.evaluation_report_writer is not None,
            'workers_ready': self.ready_workers,
            'workers_expected': self.expected_workers,
            'pending_requests': len(self._pending_predictions),
        }
        self._send_frontend(
            route,
            encode_header(
                'status_response',
                ok=True,
                request_id=str(request.get('request_id', '')),
            ), encode_frame(payload))

    def _queue_prediction(self, route: tuple[bytes, ...],
                          header: dict[str, Any], payload: bytes) -> None:
        request_id = _required_string(header, 'request_id')
        if request_id in self._predictions_by_id:
            raise ValueError(f'Duplicate active request_id {request_id!r}')
        if len(self._pending_predictions) >= self.max_pending_requests:
            if self._continuous is not None:
                raise RuntimeError('Continuous prediction queue is full')
            self._send_frontend_error(
                route,
                request_id,
                RuntimeError('prediction queue is full'),
                error_type='overloaded')
            return
        requested_deadline_ms = header.get('deadline_ms')
        timeout_s = self.request_timeout_s
        if requested_deadline_ms is not None:
            if (isinstance(requested_deadline_ms, bool)
                    or not isinstance(requested_deadline_ms, (int, float))
                    or requested_deadline_ms <= 0):
                raise ValueError('deadline_ms must be a positive number')
            timeout_s = min(timeout_s, float(requested_deadline_ms) / 1000.0)
        unnorm_key = header.get('unnorm_key', '')
        if not isinstance(unnorm_key, str):
            raise TypeError('header.unnorm_key must be a string')
        pending = _PendingPrediction(
            route=route,
            request_id=request_id,
            policy_seed=_required_int(header, 'policy_seed'),
            unnorm_key=unnorm_key,
            payload=payload,
            deadline=time.monotonic() + timeout_s,
            sampling_context=({} if self._continuous is not None else
                              _sampling_context(header)),
            request_context=(standard_request_context(header)
                             if self._continuous is not None else {}),
        )
        if self._continuous is not None:
            ticket = self._continuous.accept(route, header, payload)
            if self._fixed:
                pending.ticket = ticket
        self._pending_predictions.append(pending)
        self._predictions_by_id[request_id] = pending

    def _handle_report(self, route: tuple[bytes, ...],
                       header: Mapping[str, Any], payload: bytes) -> None:
        if self.evaluation_report_writer is None:
            raise RuntimeError('FluxVLA evaluation reporting is disabled')
        body = decode_frame(payload)
        if not isinstance(body, Mapping):
            raise TypeError('report_event payload must be a mapping')
        prepared = (
            self._continuous.prepare_report(
                _required_string(header, 'event_type'),
                _required_string(header, 'run_session_id'), body)
            if self._fixed else None)
        result = self.evaluation_report_writer.process_event(
            event_type=_required_string(header, 'event_type'),
            request_id=_required_string(header, 'request_id'),
            run_session_id=_required_string(header, 'run_session_id'),
            sequence=_required_int(header, 'sequence'),
            payload=dict(body),
        )
        if not isinstance(result, Mapping):
            raise TypeError(
                'evaluation report writer response must be a mapping')
        if self._fixed:
            if not result.get('accepted'):
                raise RuntimeError(
                    str(result.get('error') or 'Evaluation report rejected'))
            self._continuous.commit_report(prepared)
        self._send_frontend(
            route,
            encode_header(
                'report_response',
                ok=True,
                request_id=str(header.get('request_id', '')),
            ), encode_frame(dict(result)))
        accepted = bool(result.get('accepted', False))
        if not accepted or header.get('event_type') == 'run_end':
            self._begin_shutdown()

    def _receive_backend(self) -> None:
        frames = self._backend.recv_multipart()
        if len(frames) < 2:
            raise RuntimeError('model worker message is missing a header')
        identity = frames[0]
        header = decode_header(frames[1])
        self._validate_deployment(header)
        payload = frames[2] if len(frames) > 2 else empty_payload()
        message_type = header['type']
        if message_type == 'register_worker':
            self._register_worker(identity, header)
            return
        worker = self._require_worker(identity, header)
        if message_type == 'ready':
            self._mark_worker_ready(worker, header)
        elif message_type in {'predict_result', 'predict_error'}:
            self._finish_prediction(worker, header, payload)
        elif message_type == 'worker_heartbeat' and self._fixed:
            if not worker.is_ready:
                raise ValueError('Heartbeat arrived before model readiness')
            self._continuous.check_model_identity(
                worker.rank, worker.worker_id,
                header.get('process_instance_id'))
            worker.last_seen = time.monotonic()
        else:
            raise ValueError(f'Unknown model-worker message {message_type!r}')

    def _register_worker(self, identity: bytes, header: Mapping[str,
                                                                Any]) -> None:
        worker_id = _required_string(header, 'worker_id')
        if identity in self._workers or worker_id in self._worker_ids:
            raise RuntimeError(f'duplicate model worker {worker_id!r}')
        if len(self._workers) >= self.expected_workers:
            raise RuntimeError('received more model workers than expected')
        worker = _ModelWorker(
            identity=identity,
            worker_id=worker_id,
            node_id=_required_string(header, 'node_id'),
            device=_required_string(header, 'device'),
        )
        if self._fixed:
            worker.rank = _required_int(header, 'rank', minimum=0)
            if worker.rank >= self.expected_workers or any(
                    w.rank == worker.rank for w in self._workers.values()):
                raise ValueError('Duplicate or out-of-range model rank')
        self._workers[identity] = worker
        self._worker_ids.add(worker_id)

    def _mark_worker_ready(self,
                           worker: _ModelWorker,
                           header: Mapping[str, Any] | None = None) -> None:
        if worker.is_ready:
            raise RuntimeError(
                f'model worker {worker.worker_id!r} is already ready')
        if self._fixed:
            if _required_int(header or {}, 'process_sequence', minimum=0) != 0:
                raise ValueError(
                    'Continuous model must start at process sequence zero')
            self._continuous.bind_model(worker.rank, worker.worker_id,
                                        (header
                                         or {}).get('process_instance_id'),
                                        worker.device)
        elif self._continuous is not None:
            self._continuous.bind_model(worker.worker_id,
                                        (header
                                         or {}).get('process_instance_id'))
        worker.is_ready = True
        worker.last_seen = time.monotonic()
        if not self._fixed:
            self._idle_workers.append(worker.identity)
        print(
            f'[FluxVLA] model worker ready id={worker.worker_id} '
            f'node={worker.node_id} device={worker.device}',
            flush=True)

    def _finish_prediction(self, worker: _ModelWorker,
                           header: Mapping[str, Any], payload: bytes) -> None:
        request_id = _required_string(header, 'request_id')
        if (worker.prediction is None
                or worker.prediction.request_id != request_id):
            expected_request_id = (
                worker.prediction.request_id
                if worker.prediction is not None else None)
            raise RuntimeError(
                f'model worker {worker.worker_id} returned request '
                f'{request_id!r}, expected {expected_request_id!r}')
        pending = worker.prediction
        if not self._fixed:
            del self._predictions_by_id[request_id]
            self._release_worker(worker)
        if header['type'] == 'predict_error':
            error = str(header.get('error') or 'model worker failed')
            self._send_frontend_error(
                pending.route,
                request_id,
                RuntimeError(error),
                error_type='inference_error')
            if self._continuous is not None:
                self._continuous.fail(error)
                raise RuntimeError(error)
            return
        if self._continuous is not None:
            try:
                body = decode_frame(payload)
                actions = (
                    np.asarray(body.get('actions')) if isinstance(
                        body, Mapping) else None)
                elapsed = header.get('inference_time_s')
                if (actions is None or actions.ndim != 2 or 0 in actions.shape
                        or actions.dtype.kind not in 'fiu'
                        or not np.isfinite(actions).all()
                        or (np.abs(actions.astype(np.float64)) > np.finfo(
                            np.float32).max).any()):
                    raise ValueError(
                        'Continuous worker returned invalid actions')
                if (isinstance(elapsed, bool)
                        or not isinstance(elapsed, (int, float))
                        or not np.isfinite(elapsed) or elapsed < 0):
                    raise ValueError(
                        'Continuous worker returned invalid timing')
                if self._fixed:
                    self._continuous.complete(
                        request_id, worker.worker_id,
                        header.get('process_instance_id'),
                        header.get('process_sequence'))
                else:
                    self._continuous.complete(request_id)
            except Exception as exc:
                self._send_frontend_error(pending.route, request_id, exc)
                raise
        response_header = dict(header)
        if self._fixed:
            del self._predictions_by_id[request_id]
            self._release_worker(worker)
        response_header.pop('worker_id', None)
        response_header.pop('deployment_id', None)
        if self._fixed:
            response_header.pop('process_instance_id', None)
            response_header.pop('process_sequence', None)
        response_header['type'] = 'predict_response'
        response_header['ok'] = True
        self._send_frontend(pending.route, encode_frame(response_header),
                            payload)

    def _dispatch_predictions(self) -> None:
        if self._fixed:
            for _ in range(len(self._pending_predictions)):
                pending = self._pending_predictions.popleft()
                worker = next((w for w in self._workers.values()
                               if w.rank == pending.ticket.rank), None)
                if worker is None or not worker.is_idle:
                    self._pending_predictions.append(pending)
                    continue
                worker.prediction = pending
                self._send_worker(
                    'predict',
                    worker,
                    request_id=pending.request_id,
                    policy_seed=pending.policy_seed,
                    unnorm_key=pending.unnorm_key,
                    payload=pending.payload,
                    **pending.request_context)
            return
        while self._pending_predictions:
            pending = self._pending_predictions[0]
            request_id = pending.request_id
            if pending.deadline <= time.monotonic():
                self._pending_predictions.popleft()
                self._predictions_by_id.pop(request_id, None)
                self._send_frontend_error(
                    pending.route,
                    request_id,
                    TimeoutError('prediction expired while queued'),
                    error_type='timeout')
                continue
            worker = self._next_idle_worker()
            if worker is None:
                return
            self._pending_predictions.popleft()
            worker.prediction = pending
            self._send_worker(
                'predict',
                worker,
                request_id=request_id,
                policy_seed=pending.policy_seed,
                unnorm_key=pending.unnorm_key,
                payload=pending.payload,
                **pending.sampling_context,
                **pending.request_context,
            )

    def _next_idle_worker(self) -> _ModelWorker | None:
        while self._idle_workers:
            identity = self._idle_workers.popleft()
            worker = self._workers.get(identity)
            if worker is not None and worker.is_idle:
                return worker
        return None

    def _release_worker(self, worker: _ModelWorker) -> None:
        worker.prediction = None
        worker.last_seen = time.monotonic()
        if not self._fixed:
            self._idle_workers.append(worker.identity)

    def _check_idle_workers(self, now: float) -> None:
        if not self._fixed or self._shutdown_started:
            return
        for worker in self._workers.values():
            if worker.is_idle and now - worker.last_seen > 30:
                raise TimeoutError(
                    f'Idle model heartbeat expired on {worker.worker_id}')

    def _expire_predictions(self) -> None:
        now = time.monotonic()
        if self._continuous is not None:
            expired = [
                p for p in self._predictions_by_id.values()
                if p.deadline <= now
            ]
            if expired:
                error = TimeoutError('Continuous prediction timed out')
                for pending in expired:
                    self._send_frontend_error(
                        pending.route,
                        pending.request_id,
                        error,
                        error_type='timeout')
                self._continuous.fail(str(error))
                raise error
        if self._pending_predictions:
            retained = deque()
            while self._pending_predictions:
                pending = self._pending_predictions.popleft()
                request_id = pending.request_id
                if pending.deadline > now:
                    retained.append(pending)
                    continue
                self._predictions_by_id.pop(request_id, None)
                self._send_frontend_error(
                    pending.route,
                    request_id,
                    TimeoutError('prediction expired while queued'),
                    error_type='timeout')
            self._pending_predictions = retained
        for worker in list(self._workers.values()):
            pending = worker.prediction
            if pending is None:
                continue
            if pending.deadline <= now:
                request_id = pending.request_id
                self._predictions_by_id.pop(request_id, None)
                self._send_frontend_error(
                    pending.route,
                    request_id,
                    TimeoutError('model inference timed out'),
                    error_type='timeout')
                raise TimeoutError(
                    f'model inference timed out on {worker.worker_id}')

    def _require_worker(self, identity: bytes,
                        header: Mapping[str, Any]) -> _ModelWorker:
        worker = self._workers.get(identity)
        if worker is None:
            raise RuntimeError(
                'model worker must register before sending data')
        worker_id = _required_string(header, 'worker_id')
        if worker_id != worker.worker_id:
            raise RuntimeError('model worker identity mismatch')
        return worker

    def _validate_deployment(self, header: Mapping[str, Any]) -> None:
        if header.get('deployment_id') != self.deployment_id:
            raise RuntimeError('ZMQ deployment_id mismatch')

    def _send_frontend_error(self,
                             route: tuple[bytes, ...],
                             request_id: str,
                             error: Exception,
                             error_type: str = 'protocol_error') -> None:
        self._send_frontend(
            route,
            encode_header(
                'error',
                ok=False,
                request_id=request_id,
                error_type=error_type,
                error=f'{type(error).__name__}: {error}',
            ), empty_payload())

    def _send_frontend(self, route: tuple[bytes, ...], header: bytes,
                       payload: bytes) -> None:
        self._frontend.send_multipart([*route, header, payload])

    def _send_worker(self,
                     message_type: str,
                     worker: _ModelWorker,
                     payload: bytes | None = None,
                     **values: Any) -> None:
        self._backend.send_multipart([
            worker.identity,
            encode_header(
                message_type,
                deployment_id=self.deployment_id,
                worker_id=worker.worker_id,
                **values,
            ),
            payload if payload is not None else empty_payload(),
        ])

    def _begin_shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        for worker in self._workers.values():
            self._send_worker('shutdown_worker', worker)

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def stop(_signum, _frame):
            self._begin_shutdown()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)


def run_model_worker(config_path: str,
                     ckpt_path: str,
                     backend_endpoint: str,
                     worker_id: str,
                     node_id: str,
                     deployment_id: str,
                     device: str,
                     cfg_options: Mapping[str, Any] | None = None,
                     expected_workers: int = 1) -> None:
    """Load one model replica and serve independent prediction requests."""
    import zmq
    from mmengine import Config

    from fluxvla.engines.utils.torch_utils import \
        configure_inference_attention_defaults
    configure_inference_attention_defaults()

    with zmq.Context() as context:
        backend = context.socket(zmq.DEALER)
        backend.setsockopt(zmq.IDENTITY, worker_id.encode())
        backend.setsockopt(zmq.LINGER, 0)
        backend.connect(_endpoint(backend_endpoint, 'backend_endpoint'))

        common = {
            'deployment_id': deployment_id,
            'worker_id': worker_id,
        }
        cfg = Config.fromfile(config_path)
        if cfg_options:
            cfg.merge_from_dict(dict(cfg_options))
        continuous = continuous_sampling_config(
            get_server_config(
                require_mapping(config_get(cfg, 'themis'), 'themis')),
            expected_workers)
        fixed = continuous is not None and continuous.get(
            'routing') == 'fixed_worker'
        registration_context = ({
            'rank': int(worker_id.rsplit('-', 1)[1])
        } if fixed else {})
        backend.send_multipart([
            encode_header(
                'register_worker',
                **common,
                node_id=node_id,
                device=device,
                **registration_context),
            empty_payload(),
        ])
        runner = build_serving_runner_from_config(
            cfg, ckpt_path=ckpt_path, device=device)
        ready_context = ({
            'process_instance_id': runner.process_instance_id
        } if continuous is not None else {})
        if fixed:
            ready_context['process_sequence'] = runner.process_request_sequence
        backend.send_multipart([
            encode_header('ready', **common, **ready_context),
            empty_payload(),
        ])
        print(
            f'[FluxVLA] model worker ready id={worker_id} device={device}',
            flush=True)
        while True:
            if fixed and not backend.poll(1000):
                backend.send_multipart([
                    encode_header(
                        'worker_heartbeat',
                        **common,
                        process_instance_id=runner.process_instance_id),
                    empty_payload(),
                ])
                continue
            frames = backend.recv_multipart()
            header = decode_header(frames[0])
            if header.get('deployment_id') != deployment_id:
                raise RuntimeError('ZMQ deployment_id mismatch')
            if header.get('worker_id') != worker_id:
                raise RuntimeError('model worker identity mismatch')
            if header['type'] == 'shutdown_worker':
                break
            if header['type'] != 'predict':
                message_type = header['type']
                raise RuntimeError(
                    f'Unexpected model worker command {message_type!r}')
            request_id = _required_string(header, 'request_id')
            payload = frames[1] if len(frames) > 1 else empty_payload()
            try:
                result, inference_time_s, sampling_response = _predict_request(
                    runner,
                    header,
                    payload,
                    accept_standard_process=continuous is not None)
                if fixed:
                    sampling_response.update(
                        process_instance_id=runner.process_instance_id,
                        process_sequence=runner.process_request_sequence)
                backend.send_multipart([
                    encode_header(
                        'predict_result',
                        **common,
                        request_id=request_id,
                        inference_time_s=float(inference_time_s),
                        **sampling_response,
                    ),
                    encode_frame(result),
                ])
            except Exception as exc:
                print(
                    f'[FluxVLA][debug] model worker prediction failed '
                    f'worker={worker_id} request={request_id}: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )
                traceback.print_exc()
                backend.send_multipart([
                    encode_header(
                        'predict_error',
                        **common,
                        request_id=request_id,
                        error=f'{type(exc).__name__}: {exc}',
                    ),
                    empty_payload(),
                ])


def _sampling_context(header: Mapping[str, Any]) -> dict[str, Any]:
    mode = header.get('sampling_mode', 'legacy_episode')
    if mode not in {
            'legacy_episode', 'episode_noise_stream_v1',
            'process_noise_stream_v1'
    }:
        raise ValueError('Unsupported sampling_mode')
    context = {'sampling_mode': mode}
    if mode == 'process_noise_stream_v1' and 'task_id' in header:
        context['task_id'] = _required_string(header, 'task_id')
    if mode in {'episode_noise_stream_v1', 'process_noise_stream_v1'}:
        sequence = _required_int(header, 'sequence')
        if sequence < 1:
            raise ValueError('sequence must be positive')
        state = header.get('rng_state')
        if mode == 'process_noise_stream_v1' and (
                state is not None or header.get('rng_format') is not None):
            raise ValueError(
                'Process sampling does not accept caller RNG state')
        if state is not None and (not isinstance(state, bytes)
                                  or not 0 < len(state) <= 65536):
            raise ValueError('rng_state must be bounded bytes')
        context.update(
            episode_id=_required_string(header, 'episode_id'),
            sequence=sequence,
            rng_state=state,
            rng_format=header.get('rng_format'))
    return context


def _predict_request(runner,
                     header,
                     payload,
                     *,
                     accept_standard_process=False):
    """Execute the same protocol in a worker or a CPU transport test."""
    observation = decode_frame(payload)
    if not isinstance(observation, Mapping):
        raise TypeError('prediction payload must be a mapping')
    if accept_standard_process:
        standard_request_context(header)
        if runner.sampling_mode != 'process_noise_stream_v1':
            raise ValueError(
                'Standard continuous requests require process sampling')
        actions, elapsed = runner.predict_action(
            observation, str(header.get('unnorm_key', '')),
            _required_int(header, 'policy_seed'))
        return {'actions': np.asarray(actions, dtype=np.float32)}, elapsed, {}
    sampling = _sampling_context(header)
    mode = sampling['sampling_mode']
    if mode != runner.sampling_mode:
        raise ValueError('Client and model sampling_mode do not match')
    args = (observation, str(header.get('unnorm_key', '')),
            _required_int(header, 'policy_seed'))
    response = {}
    result = {}
    if mode == 'episode_noise_stream_v1':
        predict = runner.predict_action_with_sampling_state
        actions, elapsed, state, fmt = predict(*args, sampling['sequence'],
                                               sampling['rng_state'],
                                               sampling['rng_format'])
        response.update(
            sampling_mode=mode,
            episode_id=sampling['episode_id'],
            sequence=sampling['sequence'])
        result.update(rng_state=state, rng_format=fmt)
    else:
        if mode == 'process_noise_stream_v1' and getattr(
                runner, 'process_checkpoint_dir', None):
            runner.checkpoint_process_task(header.get('task_id'))
        actions, elapsed = runner.predict_action(*args)
        if mode == 'process_noise_stream_v1':
            response.update(
                sampling_mode=mode,
                episode_id=sampling['episode_id'],
                sequence=sampling['sequence'],
                process_instance_id=runner.process_instance_id,
                process_sequence=runner.process_request_sequence)
    result['actions'] = np.asarray(actions, dtype=np.float32)
    return result, elapsed, response


def build_evaluation_report_writer_from_config(
        cfg: Any,
        ckpt_path: str,
        config_path: str | os.PathLike | None = None) -> Any:
    """Build the report writer without loading a model replica."""
    themis_cfg = require_mapping(config_get(cfg, 'themis'), 'config.themis')
    server_cfg = get_server_config(themis_cfg)
    reporting_cfg = dict(
        require_mapping(
            server_cfg.get('evaluation_reporting', {}),
            'themis.server.evaluation_reporting'))
    if not bool(reporting_cfg.get('enabled', False)):
        return None
    section_name = server_cfg.get('dataset_section', 'eval')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.server.dataset_section must be `eval` or '
                         '`inference`')
    section_cfg = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    resolved_ckpt = resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    resolved_config_path = resolve_report_config_path(cfg, config_path)
    result_root = resolve_report_result_root(
        reporting_cfg.get('result_output_dir'), resolved_ckpt)
    report_eval_source = config_get(cfg, 'eval', section_cfg)
    report_eval_config = copy.deepcopy(
        dict(require_mapping(report_eval_source, 'config.eval metadata')))
    evaluation_cfg = themis_cfg.get('evaluation')
    if isinstance(evaluation_cfg, Mapping):
        if 'task_ids' in evaluation_cfg:
            report_eval_config.setdefault('task_ids',
                                          evaluation_cfg['task_ids'])
        if 'episodes_per_task' in evaluation_cfg:
            report_eval_config.setdefault('num_trials_per_task',
                                          evaluation_cfg['episodes_per_task'])
        if 'episodes_per_task_overrides' in evaluation_cfg:
            report_eval_config.setdefault(
                'num_trials_per_task_overrides',
                evaluation_cfg['episodes_per_task_overrides'])
        if 'run_name' in evaluation_cfg:
            report_eval_config.setdefault('run_name',
                                          evaluation_cfg['run_name'])
    report_eval_config.setdefault('dataset_section', section_name)
    report_eval_config.setdefault('result_gpu_id',
                                  reporting_cfg.get('result_gpu_id', 0))

    from .evaluation_report_writer import FluxVLAEvaluationReportWriter
    return FluxVLAEvaluationReportWriter(
        result_root=result_root,
        config_path=resolved_config_path,
        ckpt_path=resolved_ckpt,
        eval_config=report_eval_config,
        logger=None,
        feishu=reporting_cfg.get('feishu'),
        report_kind=reporting_cfg.get('report_kind'),
    )


def launch_server_task(*,
                       config_path: str,
                       ckpt_path: str,
                       device: str,
                       frontend_bind: str,
                       backend_bind: str,
                       backend_endpoint: str,
                       max_pending_requests: int,
                       request_timeout_s: float,
                       startup_timeout_s: float,
                       rank: int,
                       world_size: int,
                       cfg_options: Mapping[str, Any] | None = None,
                       deployment_id: str | None = None) -> None:
    """Run one torchrun model worker and the rank-zero inference server."""
    world_size = _positive_int(world_size, 'world_size')
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError('rank must be an integer')
    if rank < 0 or rank >= world_size:
        raise ValueError('rank must be in [0, world_size)')
    from mmengine import Config
    cfg = Config.fromfile(config_path, import_custom_modules=False)
    if cfg_options:
        cfg.merge_from_dict(dict(cfg_options))
    continuous = continuous_sampling_config(
        get_server_config(
            require_mapping(config_get(cfg, 'themis'), 'themis')), world_size)
    fixed = continuous is not None and continuous.get(
        'routing') == 'fixed_worker'
    if fixed:
        if int(os.environ.get('LOCAL_WORLD_SIZE', world_size)) != world_size:
            raise ValueError(
                'Fixed continuous service currently requires one node')
        deployment_id = deployment_id or os.environ.get(
            'FLUXVLA_DEPLOYMENT_ID')
        deployment_id = _nonempty(deployment_id, 'FLUXVLA_DEPLOYMENT_ID')
    elif continuous is not None:
        if continuous['audit_path'].exists():
            raise FileExistsError(continuous['audit_path'])
        deployment_id = uuid.uuid4().hex
    if world_size > 1:
        from mmengine import Config
        cfg = Config.fromfile(config_path, import_custom_modules=False)
        if cfg_options:
            cfg.merge_from_dict(dict(cfg_options))
        mode = cfg.themis.server.get('runner', {}).get('sampling_mode')
        if mode == 'process_noise_stream_v1' and not fixed:
            raise ValueError(
                'Process sampling requires one model worker per endpoint')
    node_id = f'{network_socket.gethostname()}-rank-{rank}'
    if deployment_id is None:
        deployment_id = (
            os.environ.get('FLUXVLA_DEPLOYMENT_ID')
            or os.environ.get('TORCHELASTIC_RUN_ID'))
    if not deployment_id:
        stable = '{}:{}:{}:{}'.format(
            os.environ.get('MASTER_ADDR', '127.0.0.1'),
            os.environ.get('MASTER_PORT', '29500'),
            Path(config_path).resolve(),
            Path(ckpt_path).resolve(),
        )
        deployment_id = hashlib.sha256(stable.encode()).hexdigest()[:32]

    server_process = None
    server_watcher = None
    if rank == 0:
        context = multiprocessing.get_context('spawn')
        server_process = context.Process(
            target=_server_process_main,
            kwargs={
                'config_path': config_path,
                'ckpt_path': ckpt_path,
                'cfg_options': dict(cfg_options or {}),
                'frontend_bind': frontend_bind,
                'backend_bind': backend_bind,
                'expected_workers': world_size,
                'max_pending_requests': max_pending_requests,
                'request_timeout_s': request_timeout_s,
                'startup_timeout_s': startup_timeout_s,
                'deployment_id': deployment_id,
                'parent_pid': os.getpid(),
            },
            name='fluxvla-inference-server',
        )
        _start_server_process(server_process)
        server_watcher = threading.Thread(
            target=_terminate_on_server_failure,
            args=(server_process, os.getpid()),
            daemon=True,
        )
        server_watcher.start()

    try:
        run_model_worker(
            config_path=config_path,
            ckpt_path=ckpt_path,
            backend_endpoint=backend_endpoint,
            worker_id=f'model-worker-{rank}',
            node_id=node_id,
            deployment_id=deployment_id,
            device=device,
            cfg_options=cfg_options,
            expected_workers=world_size,
        )
    except BaseException:
        if server_process is not None:
            if server_process.is_alive():
                server_process.terminate()
            server_watcher.join()
        raise
    else:
        if server_watcher is not None:
            server_watcher.join()


def _start_server_process(process: multiprocessing.Process) -> None:
    """Start the inference server outside the model workers' ProcessGroup."""
    world_size = os.environ.pop('WORLD_SIZE', None)
    try:
        process.start()
    finally:
        if world_size is not None:
            os.environ['WORLD_SIZE'] = world_size


def _terminate_on_server_failure(process: multiprocessing.Process,
                                 parent_pid: int) -> None:
    process.join()
    if process.exitcode not in {0, -signal.SIGTERM}:
        print(
            f'[FluxVLA] inference server exited with code {process.exitcode}',
            file=sys.stderr,
            flush=True,
        )
        os.kill(parent_pid, signal.SIGTERM)


def _server_process_main(*, config_path: str, ckpt_path: str,
                         cfg_options: Mapping[str, Any], frontend_bind: str,
                         backend_bind: str, expected_workers: int,
                         max_pending_requests: int, request_timeout_s: float,
                         startup_timeout_s: float, deployment_id: str,
                         parent_pid: int) -> None:
    _configure_child_lifetime(parent_pid)
    from mmengine import Config
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(dict(cfg_options))
    report_writer = build_evaluation_report_writer_from_config(
        cfg, ckpt_path=ckpt_path, config_path=config_path)
    server = FluxVLAInferenceServer(
        evaluation_report_writer=report_writer,
        frontend_bind=frontend_bind,
        backend_bind=backend_bind,
        expected_workers=expected_workers,
        max_pending_requests=max_pending_requests,
        request_timeout_s=request_timeout_s,
        startup_timeout_s=startup_timeout_s,
        deployment_id=deployment_id,
        continuous_sampling=continuous_sampling_config(
            get_server_config(
                require_mapping(config_get(cfg, 'themis'), 'themis')),
            expected_workers),
    )
    server.run()


def _configure_child_lifetime(parent_pid: int) -> None:
    if sys.platform != 'linux':
        return
    import ctypes
    expected = int(parent_pid)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected:
        os.kill(os.getpid(), signal.SIGKILL)


def _endpoint(value: Any, name: str) -> str:
    value = _nonempty(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


def _required_string(header: Mapping[str, Any], name: str) -> str:
    return _nonempty(header.get(name), f'header.{name}')


def _required_int(header: Mapping[str, Any],
                  name: str,
                  minimum: int | None = None) -> int:
    value = header.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'header.{name} must be an integer')
    if minimum is not None and value < minimum:
        raise ValueError(f'header.{name} must be >= {minimum}')
    return int(value)


__all__ = [
    'FluxVLAInferenceServer',
    'build_evaluation_report_writer_from_config',
    'launch_server_task',
    'run_model_worker',
]
