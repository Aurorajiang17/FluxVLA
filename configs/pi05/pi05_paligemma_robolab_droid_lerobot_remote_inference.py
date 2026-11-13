"""Explicit continuous deployment for ordinary FluxThemis requests."""
import os

_base_ = './pi05_paligemma_robolab_droid_lerobot_inference.py'
themis = dict(
    server=dict(
        continuous_sampling=dict(
            enabled=True,
            audit_path=os.environ.get('FLUXVLA_CONTINUOUS_AUDIT_PATH'),
            max_audit_bytes=134217728,
        )))
