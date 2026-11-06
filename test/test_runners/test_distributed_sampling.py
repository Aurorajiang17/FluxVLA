from collections import deque
from test.test_runners.test_sampling_stream import make_runner

import numpy as np
import pytest
import torch

from fluxvla.engines.runners.serving.distributed_server import (
    FluxVLAInferenceServer, _ModelWorker, _predict_request, _sampling_context)
from fluxvla.engines.runners.serving.zmq_protocol import (decode_frame,
                                                          decode_header,
                                                          encode_frame,
                                                          encode_header)


def test_broker_forwards_sampling_state_across_two_workers():
    server = FluxVLAInferenceServer(
        'tcp://127.0.0.1:5101', 'tcp://127.0.0.1:5102', expected_workers=2)
    workers = [
        _ModelWorker(bytes([i]), str(i), 'node', 'cpu', True) for i in (1, 2)
    ]
    server._workers = {w.identity: w for w in workers}
    server._idle_workers = deque(w.identity for w in workers)
    sent, received, used = [], [], []
    server._send_worker = lambda kind, worker, payload, **header: sent.append(
        (worker, decode_header(encode_header(kind, **header)), payload))
    server._send_frontend = lambda route, header, payload: received.append(
        (decode_header(header), decode_frame(payload)))
    runners = {w.identity: make_runner() for w in workers}
    torch.manual_seed(57)
    torch.rand(3)
    references = [torch.normal(0., 1., size=(3, 8)).numpy() for _ in range(6)]
    state = fmt = None
    for sequence, reference in enumerate(references, 1):
        request = dict(
            request_id=str(sequence),
            policy_seed=57,
            episode_id='episode',
            sequence=sequence,
            sampling_mode='episode_noise_stream_v1',
            rng_state=state,
            rng_format=fmt)
        server._queue_prediction((b'client', ), request, encode_frame({}))
        server._dispatch_predictions()
        worker, header, payload = sent.pop()
        used.append(worker.identity)
        body, elapsed, extra = _predict_request(runners[worker.identity],
                                                header, payload)
        server._finish_prediction(
            worker,
            decode_header(
                encode_header(
                    'predict_result',
                    request_id=str(sequence),
                    inference_time_s=elapsed,
                    **extra)), encode_frame(body))
        response, body = received.pop()
        assert response['episode_id'] == 'episode'
        assert response['sequence'] == sequence
        np.testing.assert_array_equal(body['actions'], reference)
        state, fmt = body['rng_state'], body['rng_format']
    assert used == [workers[i % 2].identity for i in range(6)]


@pytest.mark.parametrize('change', [
    dict(sequence=True),
    dict(sequence=0),
    dict(episode_id=''),
    dict(rng_state=b'x' * 65537),
    dict(sampling_mode='unknown')
])
def test_bad_protocol_metadata_is_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        _sampling_context(
            dict(
                sampling_mode='episode_noise_stream_v1',
                episode_id='a',
                sequence=1,
                **{}) | change)


def test_mode_mismatch_cannot_fall_back_to_reseeding():
    with pytest.raises(ValueError, match='do not match'):
        _predict_request(make_runner(), dict(policy_seed=57), encode_frame({}))
