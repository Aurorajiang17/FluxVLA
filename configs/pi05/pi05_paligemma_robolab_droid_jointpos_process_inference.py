"""RoboLab recipe with one continuous default RNG stream per model process.

Use one persistent model worker per endpoint and pair it with one simulator
worker. The fixed start seed is a deployment choice; request order determines
which noise each episode receives. Weight and compute settings are inherited.
"""

_base_ = './pi05_paligemma_robolab_droid_jointpos_inference.py'

themis = dict(
    server=dict(
        runner=dict(sampling_mode='process_noise_stream_v1', process_seed=0),
        torch_runtime=dict(
            num_threads=2,
            num_interop_threads=1,
            float32_matmul_precision='highest',
            allow_tf32_matmul=False,
            allow_tf32_cudnn=False)))
