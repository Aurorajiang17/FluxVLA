# FluxVLA Serving

For ordinary FluxThemis single-environment RoboLab evaluation, use the
[RoboLab remote evaluation guide](robolab_remote_evaluation.md). It uses the
distributed server; the real-robot protocol and explicit process recipes below
retain their own entry points.
The distributed server also offers opt-in continuous_sampling.routing='fixed_worker':
N persistent model instances serve N fixed prediction clients through one broker.
It requires single-node fixed assignment and complete evaluation reporting.
The RoboLab guide owns GPU-list selection, configuration and launch commands;
default dynamic serving and the existing single_client mode remain available.

Process-mode runners optionally accept `process_checkpoint_dir` to save the
actual Python, NumPy, CPU Torch and device Torch states before a task's first
request. `process_resume=dict(path=..., sha256=...)` explicitly restores a
verified state and its request count after model initialization. The seed,
Torch version and device type must match. Each restored service has a new
physical process identity, so evaluation reports must retain the old lineage.
These options do not assume any particular model dimensions or noise sampler.

FluxVLA Serving provides a lightweight, high-performance remote inference
framework that decouples VLA model execution from robot control. A GPU server
hosts the model behind a ZMQ (ZeroMQ) REP socket, while the robot-side client
sends observations and receives predicted actions over a simple
request-reply protocol — requiring only `pyzmq`, `msgpack`, and `numpy` on
the robot with no GPU or deep-learning stack needed.

Two wire formats are supported (msgpack and protobuf), observations are
JPEG-compressed on the wire for bandwidth efficiency, and an optional SSH
tunnel mode allows seamless deployment when the GPU machine sits behind NAT
or in the cloud.

## Architecture

```
serving/
├── serve.py          # CLI entry point: load model + start server
├── zmq_server.py     # PolicyServer (ZMQ event loop) + create_server factory
├── serializers.py    # Wire-format encode/decode (shared by client & server)
└── proto/
    ├── vla_service.proto      # Protobuf schema
    └── vla_service_pb2.py     # Generated Python code
```

Two-layer design:

```
PolicyServer        Layer 1: generic ZMQ REP loop + endpoint routing
    |
create_server()     Layer 2: wires VLA model into predict_action handler
                    (deserialize obs -> preprocess -> inference ->
                     denormalize -> serialize action)
```

## Environment Setup

### Server side (GPU machine)

Requires GPU + CUDA. Install the full FluxVLA training environment.

### Client side (robot machine, no GPU needed)

Minimal Python dependencies:

```
pyzmq>=27.0
msgpack>=1.0
numpy
```

If using `serializer='protobuf'`, also install:

```
protobuf>=5.26
```

**Note:** `protobuf` version must be compatible with `wandb` if wandb is also
installed. If you hit import errors, either:

- Upgrade both: `pip install --upgrade protobuf wandb`
- Or use `serializer='msgpack'` (no protobuf dependency, recommended for
  robot-side environments with pinned packages)

## Quick Start

### Step 1: Start the server (GPU machine)

```bash
python -m fluxvla.engines.runners.serving.serve \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 3333 \
    --device cuda:0 --dtype bf16
```

The server loads the model, dataset pipeline, and denormalization transform,
then starts a ZMQ REP loop waiting for requests.

### Step 2: Set up network connection

If the robot cannot reach the GPU server directly (e.g. server is behind NAT
or on a cloud machine), use an SSH tunnel to forward a local port to the
server's ZMQ port:

```bash
ssh -p <SSH_PORT> -i <KEY_FILE> \
    -L <LOCAL_PORT>:127.0.0.1:<SERVER_ZMQ_PORT> \
    <USER>@<SERVER_PUBLIC_IP> -N
```

Example: server runs ZMQ on port 3333, forward local 5555 to it:

```bash
ssh -p 57705 -i ~/.ssh/my_key \
    -L 5555:127.0.0.1:3333 \
    user@server.example.com -N
```

Explanation:

- `-L 5555:127.0.0.1:3333`: binds local port 5555, tunnels traffic to
  `127.0.0.1:3333` on the remote server (where ZMQ is listening)
- `-N`: no remote shell, just forwarding
- `-p 57705`: SSH port if non-standard (default is 22)

After this, the robot connects to `127.0.0.1:5555` and traffic is tunneled
to the GPU server's ZMQ port.

If the robot and GPU server are on the same LAN, skip the tunnel and point
`server_host` directly to the server's LAN IP.

### Step 3: Create client config

Add `remote_inference` to any existing runner config.  The runner type
stays the same (`URInferenceRunner`, `AlohaInferenceRunner`, etc.):

```python
# configs/pi05/pi05_paligemma_ur3_remote_inference.py
inference = dict(
    type='URInferenceRunner',          # same runner as local inference
    remote_inference=dict(              # add this block to enable remote mode
        server_host='127.0.0.1',       # localhost if using SSH tunnel
        server_port=5555,
        timeout_s=30.0,
        serializer='msgpack',          # 'msgpack' (recommended) or 'protobuf'
        compress=True,
        enable_profiling=True,
    ),
    seed=7,
    action_chunk=10,
    publish_rate=30,
    max_publish_step=10000,
    task_suite_name='private',
    state_dim=7,
)
```

For Aloha, just change `type='AlohaInferenceRunner'`.

### Step 4: Run the client (robot)

Option A: Direct launch

```bash
bash scripts/remote_inference_client.sh \
    configs/pi05/pi05_paligemma_ur3_remote_inference.py \
    --ssh-host user@server.example.com \
    --ssh-port 57705 \
    --ssh-key ~/.ssh/my_key \
    --local-port 5555 \
    --remote-port 3333
```

Option B: One-command launch with auto SSH tunnel

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
```

The script automatically starts the SSH tunnel, runs inference, and cleans
up the tunnel on exit.  SSH parameters:

- `--ssh-host`: SSH destination (user@host), enables auto tunnel
- `--ssh-port`: SSH port (default 22)
- `--ssh-key`: path to private key file
- `--local-port`: local tunnel entry port (default 5555)
- `--remote-port`: remote ZMQ server port (default 3333)

This will:

1. Build a `URInferenceRunner` with `remote_inference` enabled
2. `run_setup()` connects to the ZMQ server and sends a ping health-check
3. `run()` enters the ROS inference loop:
   collect obs -> encode -> ZMQ send -> receive -> decode action -> execute

Note: `--ckpt-path` is **not needed** for remote inference (the server loads
the model).

### Runtime interaction

During inference, the terminal prompts for task selection:

- Enter a task ID (e.g. `1`, `2`, ...) to run a predefined task
- Enter `0` to reset the robot to its prepare pose (local ROS, no network)
- Enter the number of times to repeat the task

### Supported runners

Any runner that inherits from `BaseInferenceRunner` supports remote mode
by adding `remote_inference` to its config:

| Runner                    | Robot                |
| ------------------------- | -------------------- |
| `URInferenceRunner`       | UR3/UR5 single-arm   |
| `AlohaInferenceRunner`    | Aloha dual-arm       |
| `AlohaRTCInferenceRunner` | Aloha dual-arm + RTC |

### Client config options (remote_inference dict)

| Option             | Default     | Description                          |
| ------------------ | ----------- | ------------------------------------ |
| `server_host`      | `localhost` | GPU server IP or hostname            |
| `server_port`      | `5555`      | ZMQ TCP port                         |
| `timeout_s`        | `30.0`      | ZMQ send/recv timeout (seconds)      |
| `serializer`       | `msgpack`   | Wire format: `msgpack` or `protobuf` |
| `compress`         | `True`      | JPEG-compress images before sending  |
| `enable_profiling` | `True`      | Print avg latency every 50 calls     |

## CLI Arguments

| Argument        | Default     | Description                                            |
| --------------- | ----------- | ------------------------------------------------------ |
| `--config`      | (required)  | Path to mmengine config file                           |
| `--ckpt-path`   | (required)  | Path to model checkpoint (.pt / .safetensors)          |
| `--host`        | `0.0.0.0`   | Bind address (`0.0.0.0` = all interfaces)              |
| `--port`        | `5555`      | ZMQ TCP port                                           |
| `--device`      | `cuda:0`    | CUDA device                                            |
| `--dtype`       | `bf16`      | Mixed precision: `bf16` / `fp16` / `fp32`              |
| `--dataset-key` | auto-detect | Config key for dataset pipeline: `inference` or `eval` |

## Wire Formats

Two serialization formats are supported. The first byte of every ZMQ message
determines the format:

| Format   | First byte | Pros                                  | Cons                     |
| -------- | ---------- | ------------------------------------- | ------------------------ |
| msgpack  | != `0x01`  | Flexible, zero schema, Python-native  | Larger payload           |
| protobuf | `0x01`     | Compact, cross-language (C++/Rust/Go) | Requires `.proto` schema |

Both formats use the same payload encoding for observation data:

- RGB images: JPEG-compressed bytes (configurable via `compress` flag)
- Numeric arrays: numpy `.npy` format bytes
- Strings: passed directly

## Endpoints

| Endpoint         | Input            | Output                             | Description        |
| ---------------- | ---------------- | ---------------------------------- | ------------------ |
| `predict_action` | obs + unnorm_key | action bytes + infer_time          | Model inference    |
| `ping`           | (none)           | `{status: ok}`                     | Health check       |
| `reset`          | (none)           | `{status: ok}`                     | Reset server state |
| `get_status`     | (none)           | uptime, request count, avg latency | Server stats       |
| `kill`           | (none)           | `{status: ok}`                     | Graceful shutdown  |

## Data Flow

```
Client (robot)                    Server (GPU)
     |                                 |
     |  encode obs (JPEG + npy)        |
     |  ────── ZMQ REQ ──────────────> |
     |                                 |  deserialize obs
     |                                 |  dataset transform (preprocess)
     |                                 |  vla.predict_action (GPU)
     |                                 |  denormalize action
     |                                 |  serialize action (npy)
     |  <────── ZMQ REP ────────────── |
     |  decode action                  |
     |  execute on robot               |
```

## RoboLab process sampling

The matching FluxThemis entry `scripts/run_robolab_process.py` launches this
repository's public `scripts/distributed_zmq_server.py` with
`configs/pi05/pi05_paligemma_robolab_droid_jointpos_process_inference.py`.
One persistent model serves each complete logical task stream. Themis can
schedule independent streams across one or more physical GPUs.

The process recipe inherits checkpoint loading, parameter dtypes, BF16
autocast, preprocessing and action conversion from the DROID episode recipe.
Its explicit `torch_runtime` disables TF32, selects highest float32 matmul
precision, and uses two model threads and one interop thread. These settings
apply before model loading and remain optional for other serving recipes.

`process_seed=0` seeds Python, NumPy and Torch once after model initialization.
Subsequent requests consume the default continuous device stream across lanes,
episodes and tasks. Request episode seeds do not reseed it. This follows
OpenPI's PyTorch stream lifecycle; OpenPI does not mandate the integer 0.
Responses carry a process instance ID and increasing request ordinal.

Use one model worker per endpoint, independent endpoints for additional
streams, fixed task/request order and no warmup. Keep each service alive until
its complete task list ends. Model failures seal the stream; recovery needs
the exact state and prefix, not reseeding and skipping completed tasks.

The original `pi05_paligemma_robolab_droid_jointpos_inference.py` remains the
explicit episode recipe with existing multiworker behavior. Both sides must
select the same protocol. Install matching Themis client/launcher changes
before selecting process mode.

Assets remain explicit through `FLUXVLA_ROBOLAB_CHECKPOINT`,
`FLUXVLA_ROBOLAB_STATS` and `FLUXVLA_ROBOLAB_TOKENIZER`. Prepared matching assets
are required. No private diagnostic adapter, `sitecustomize.py` or experimental
model service wrapper is needed by the public entry.

## RoboLab models trained with the DROID LeRobot recipe

Checkpoints from `configs/pi05/pi05_paligemma_droid_lerobot_full_finetune.py`
are served with `configs/pi05/pi05_paligemma_robolab_droid_lerobot_inference.py`.
It inherits the process recipe above and differs only in two ways: the model
returns 16-step chunks, its training horizon, and assets come from
`FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT`, `FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER` and
`FLUXVLA_ROBOLAB_DROID_FT_STATS`. Without the statistics variable the server
reads the training run's `dataset_statistics.json` next to `checkpoints/`.
The released checkpoint keeps the 15-step jointpos recipes.

The number of returned actions executed before the next observation is an
evaluation setting (Themis `execute_horizon`), not part of this config.
Evaluate these checkpoints executing 15 of the 16 actions, the replanning
interval of the released pi05 baseline. Themis `configs/robolab/droid.py` uses
`execute_horizon=None`, which executes all 16; point the process launch file's
`simulator_config` at a config that inherits it with `execute_horizon=15`.
