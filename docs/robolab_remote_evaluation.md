# RoboLab remote evaluation

Run simulation and scheduling through FluxThemis and start this service separately.
The default recipe uses one model worker and one prediction client. The optional
fixed mode shares one endpoint across N model workers and N prediction clients.
This guide uses ordinary FluxThemis requests through the distributed server;
the real-robot protocol is in [FluxVLA Serving](remote_inference_serving.md).

## Installation

Create an isolated Python 3.10 environment from a compatible FluxVLA checkout.
Prepare matching NVIDIA drivers and CUDA toolkit. The existing
[installer](../scripts/install_env.sh) owns package versions and normally builds
the three CUDA extensions from [setup.py](../setup.py).
For a RoboLab-only model host on CUDA 12.8, run from the FluxVLA root:

```bash
python3.10 -m venv /path/to/model-env
export MODEL_PYTHON=/path/to/model-env/bin/python
export PATH="/path/to/model-env/bin:$PATH"
export PYTHON="$MODEL_PYTHON"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
"$MODEL_PYTHON" -m pip install --upgrade 'pip>=24.2,<27'
"$MODEL_PYTHON" -m pip install 'cmake<4' ninja
export PATH="$("$MODEL_PYTHON" -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"
cmake --version
ninja --version
"$MODEL_PYTHON" -m pip install 'protobuf>=3.20.3,<5' \
  google-api-core proto-plus googleapis-common-protos
bash scripts/install_env.sh sim-only --profile cu128 \
  --skip-build-tools --skip-robocasa --skip-robotwin --skip-egl-setup
"$MODEL_PYTHON" -m pip check
```

The version commands verify the manually installed tools. --skip-build-tools
avoids the installer's Conda fallback for pip CMake wrappers. The native CMake
path also works inside isolated dependency builds; the installer still builds
FluxVLA's extensions normally.
Choose the profile for the host. TORCH_CUDA_ARCH_LIST may explicitly select
its actual GPU architecture. Skipped simulator setup belongs to the separate
FluxThemis host. Do not copy old extension binaries or add private import paths.
The model and Isaac hosts use separate Python environments. TensorFlow 2.15
simulation dependencies require protobuf below 5; resolve the Google clients
with that constraint to keep the base and simulator dependencies compatible.
The installer's final model-registry import requires a visible CUDA GPU.

## Matching assets and configuration

Use [jointpos_remote_inference.py](../configs/pi05/pi05_paligemma_robolab_droid_jointpos_remote_inference.py)
with matching pi05_droid_jointpos PyTorch weights, PaliGemma tokenizer and
official DROID statistics from [OpenPI](https://github.com/Physical-Intelligence/openpi)
and [RoboLab](https://github.com/NVlabs/RoboLab). Converted weights must retain
the recipe's names, shapes and dtypes; the loader accepts PyTorch/safetensors.
Supply the assets explicitly:

```bash
export FLUXVLA_ROBOLAB_CHECKPOINT=/path/to/model.safetensors
export FLUXVLA_ROBOLAB_TOKENIZER=/path/to/paligemma_tokenizer.model
export ROBOLAB_OFFICIAL_STATS=/path/to/droid/norm_stats.json
export FLUXVLA_ROBOLAB_STATS=/path/to/new/dataset_statistics.json
"$MODEL_PYTHON" - <<'PYTHON'
import json, os
from pathlib import Path
raw = json.loads(Path(os.environ['ROBOLAB_OFFICIAL_STATS']).read_text())['norm_stats']
mapped = {'robolab_droid': {}}
for source, target in [('state', 'state'), ('actions', 'action')]:
    mapped['robolab_droid'][target] = {}
    for field in ['mean', 'std', 'q01', 'q99']:
        values = raw[source][field]
        if len(values) < 8:
            raise ValueError(f'{source}.{field}: fewer than eight values')
        mapped['robolab_droid'][target][field] = values[:8]
with Path(os.environ['FLUXVLA_ROBOLAB_STATS']).open('x') as file:
    json.dump(mapped, file, indent=2)
PYTHON
```

The destination's parent must exist. Mapping preserves the official source file.
The released recipe returns fifteen actions and retains its FP32 checkpoint,
mixed parameter containers and BF16 autocast settings.

For weights from the [DROID training recipe](../configs/pi05/pi05_paligemma_droid_lerobot_full_finetune.py),
use [lerobot_remote_inference.py](../configs/pi05/pi05_paligemma_robolab_droid_lerobot_remote_inference.py).
Set FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT, FLUXVLA_ROBOLAB_DROID_FT_TOKENIZER
and FLUXVLA_ROBOLAB_DROID_FT_STATS to the matching training-run assets.
These statistics already use the FluxVLA schema; do not apply the released
model's mapping or substitute its statistics.
Replace MODEL_CONFIG and MODEL_CHECKPOINT in the next block with the LeRobot
recipe and "$FLUXVLA_ROBOLAB_DROID_FT_CHECKPOINT".
For the H16 LeRobot recipe, the model returns sixteen actions;
FluxThemis droid_h16.py executes fifteen before replanning. The base droid.py
executes the full returned chunk.

For a bundle produced by the training checkpoint export tool, use its config.py
as MODEL_CONFIG and model.safetensors as MODEL_CHECKPOINT. Keep the bundled
tokenizer and dataset_statistics.json together. Inspect training_metadata.json
for training_horizon and deployment_horizon; the exported deployment configuration
is authoritative. Native FluxVLA keys use strict loading with no conversion
mapping. Set the continuous-sampling options below explicitly for an exported
configuration; exporting does not select the sampling policy. Serving changes
do not require retraining existing weights.

## Freeze and start

From the FluxVLA root, choose persistent run/result directories and a new audit
file. FluxThemis supplies actual task and episode quotas through reporting.

Choose a mode before freezing. Single-card use keeps the existing contract:

```bash
export GPU_IDS=0
export CONTINUOUS_ROUTING=single_client
```

For N cards on one node, use fixed scheduling and one model per simulator
worker. Replace the GPU list with available device IDs or full GPU UUIDs:

```bash
export GPU_IDS=0,1
export CONTINUOUS_ROUTING=fixed_worker
```

Then execute the common setup from the FluxVLA root. The GPU list determines N;
do not separately choose a different worker count:

```bash
export RUN_DIRECTORY=/path/to/new/run
mkdir -p "$RUN_DIRECTORY/model-results"
export GPU_COUNT="$("$MODEL_PYTHON" - <<'PYTHON'
import os
ids = [item.strip() for item in os.environ['GPU_IDS'].split(',')]
if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
    raise ValueError('GPU_IDS must contain distinct nonempty IDs')
if os.environ['CONTINUOUS_ROUTING'] not in {'single_client', 'fixed_worker'}:
    raise ValueError('Unsupported CONTINUOUS_ROUTING')
if os.environ['CONTINUOUS_ROUTING'] == 'single_client' and len(ids) != 1:
    raise ValueError('single_client requires one GPU')
print(len(ids))
PYTHON
)"
export FLUXVLA_DEPLOYMENT_ID="$("$MODEL_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)')"
export FLUXVLA_CONTINUOUS_AUDIT_PATH="$RUN_DIRECTORY/continuous-audit.jsonl"
# Released model; replace these two assignments for fine-tuned weights.
export MODEL_CONFIG=configs/pi05/pi05_paligemma_robolab_droid_jointpos_remote_inference.py
export MODEL_CHECKPOINT="$FLUXVLA_ROBOLAB_CHECKPOINT"
export FROZEN_MODEL_CONFIG="$RUN_DIRECTORY/model.py"
export MODEL_RESULT_DIRECTORY="$RUN_DIRECTORY/model-results"
"$MODEL_PYTHON" - <<'PYTHON'
import os
from pathlib import Path
from mmengine import Config
out = Path(os.environ['FROZEN_MODEL_CONFIG'])
if out.exists():
    raise FileExistsError(out)
cfg = Config.fromfile(os.environ['MODEL_CONFIG'], lazy_import=False)
cfg.themis.server.runner.update(
    sampling_mode='process_noise_stream_v1', process_seed=0)
cfg.themis.server.continuous_sampling = dict(
    enabled=True, routing=os.environ['CONTINUOUS_ROUTING'],
    audit_path=os.environ['FLUXVLA_CONTINUOUS_AUDIT_PATH'])
cfg.themis.server.evaluation_reporting.update(
    enabled=True, result_output_dir=os.environ['MODEL_RESULT_DIRECTORY'])
cfg.dump(str(out))
PYTHON
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$MODEL_PYTHON" -m torch.distributed.run --standalone \
  --nproc-per-node="$GPU_COUNT" --max-restarts=0 scripts/distributed_zmq_server.py \
  --config "$FROZEN_MODEL_CONFIG" --ckpt-path "$MODEL_CHECKPOINT" \
  --frontend-bind tcp://127.0.0.1:15555 \
  --backend-bind tcp://127.0.0.1:15556 \
  --backend-endpoint tcp://127.0.0.1:15556
```

This example binds to localhost and explicitly routes model workers to that
backend. torchrun standalone may set MASTER_ADDR to the host address, so the
backend endpoint must be explicit for a loopback-only bind.
For separate hosts, bind the frontend to a
reachable model-host interface and configure that endpoint in FluxThemis.
See the shared evaluation networking guide for firewall/tunnel setup.
Every run needs a new frozen configuration and audit path.

For fixed mode, all ranks inherit the same FLUXVLA_DEPLOYMENT_ID from this shell.
Rank zero owns the broker and audit file; each rank owns one model instance.
CUDA_VISIBLE_DEVICES maps the chosen list to local cuda:0 through cuda:N-1.
Use the same GPU list, N and endpoint for FluxThemis. In the FluxThemis root,
with its simulator installed and ROBOLAB_ROOT/ROBOLAB_PYTHON set as in its guide:

```bash
export FLUXTHEMIS_PYTHON="$ROBOLAB_PYTHON"
# Set these to the same values chosen above, including in another terminal.
export GPU_IDS=0,1
export RUN_DIRECTORY=/path/to/new/run
export FROZEN_THEMIS_YAML="$RUN_DIRECTORY/themis.yaml"
"$FLUXTHEMIS_PYTHON" - <<'PYTHON'
import os
from pathlib import Path
import yaml
ids = [item.strip() for item in os.environ['GPU_IDS'].split(',')]
if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
    raise ValueError('GPU_IDS must contain distinct nonempty IDs')
gpu_ids = [int(item) if item.isdecimal() else item for item in ids]
cfg = yaml.safe_load(Path('configs/robolab/remote.yaml').read_text())
# Preserve relative config lookup after moving the YAML outside configs/robolab.
cfg['simulator'].update(config=str(Path('configs/robolab/droid.py').resolve()),
                        gpu_ids=gpu_ids, workers_per_gpu=1, scheduling='fixed',
                        simulator_restart_limit=0)
cfg['model'].update(expected_workers=len(ids), host='127.0.0.1', port=15555)
cfg['run']['output_root'] = str(Path(os.environ['RUN_DIRECTORY']) / 'themis-results')
cfg['resume']['from_dir'] = None
with Path(os.environ['FROZEN_THEMIS_YAML']).open('x') as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PYTHON
bash scripts/run_evaluation.sh --config "$FROZEN_THEMIS_YAML" --dry-run
bash scripts/run_evaluation.sh --config "$FROZEN_THEMIS_YAML" --full
```

The --full flag selects the full task suite. For a smaller run, start without
--full or --smoke so the frozen task selection and episode quotas are preserved:

```bash
bash scripts/run_evaluation.sh --config "$FROZEN_THEMIS_YAML"
```

For one task, set simulator.task and simulator.episodes in the YAML. For a task
list, set environment.task_name and episodes_per_task in the Python evaluation
configuration and keep simulator.task and simulator.episodes null in the YAML.
Match the Python config to the model's deployment horizon. A smoke run requires
a separate fresh service. The model service remains outside
FluxThemis; no per-card model launcher is added to the benchmark.

## Continuous sampling contract and results

The remote recipes explicitly enable continuous_sampling; other configurations
retain their existing sampling behavior. The model seeds the existing process
stream once after loading, then consumes it across episodes and tasks.
Ordinary policy_seed metadata does not reseed it. No caller RNG state or
sampling-mode fields are accepted.

In single_client mode, one client owns the endpoint and requests run sequentially.
In fixed_worker mode, each client permanently owns one model rank. The existing
run_start manifest must declare fixed assignment, N workers and full episode
quotas. Global task-major episode numbers start at one; rank is
(episode_number - 1) % N. For two tasks with four episodes each and N=2,
each rank runs two episodes of each task. Within a worker, each task's
environment is reused; switching tasks creates a new environment. Thus a
task can have one environment instance per participating worker. Native
instance_episode counts local reuse, rather than the global episode index.

Each model seeds once and continues its own process stream across tasks.
Equal seeds can produce equal initial random sequences; these streams are
not asserted to be statistically independent. Changing N or the task order
changes which inputs consume each stream and can change actions and scores.
Keep N, assignment, task order, seed, weights and action horizon fixed when
comparing checkpoints. Additional cards do not guarantee N-fold acceleration.

Request IDs cannot
repeat, sequences advance by one, and every new episode begins at one.
Invalid requests, prediction errors, timeouts, model replacement and audit
failure seal the service. Disable simulator retries and resume.
Status and result reporting do not invoke prediction.
Fixed mode requires complete reporting and permits one run per deployment.
Status/report connections do not claim prediction routes. Any rank's technical
failure seals the entire service; a completed rank does not stop the other ranks.
Idle model heartbeats do not consume random numbers. Dynamic routing, restart,
resume and multiple nodes are unsupported in fixed mode. Other service modes
keep their existing behavior.

The exclusively created audit records deployment/model identity, configuration
fingerprint, accepted/completed requests and payload hashes, bounded to 128 MiB.
Compare its request IDs with FluxThemis episode prediction_metadata.
An accepted request without completion makes the evaluation incomplete.
Fixed-mode audit records include rank, global acceptance order, each rank's
stream sequence and verified model process identity/count. Completion order
may differ from acceptance order. Internal process fields stay between model
workers and the broker; ordinary FluxThemis requests remain unchanged.
Reporting stores native success and Score separately in MODEL_RESULT_DIRECTORY.
Output quotas and task scheduling come from FluxThemis.

Smoke and full need separate fresh deployments and audit files. Do not use a
smoke-consumed stream as full evaluation's seed-zero start.
Single-environment ordering differs from historical batched ordering;
continuous sampling does not imply identical historical actions or scores.
