from pathlib import Path

import pytest
from mmengine import Config


@pytest.mark.parametrize('stem,base', [
    ('jointpos', 'pi05_paligemma_robolab_droid_jointpos_process_inference.py'),
    ('lerobot', 'pi05_paligemma_robolab_droid_lerobot_inference.py')
])
def test_remote_recipe_only_adds_explicit_service_contract(
        stem, base, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2] / 'configs/pi05'
    monkeypatch.setenv('FLUXVLA_CONTINUOUS_AUDIT_PATH',
                       str(tmp_path / 'audit.jsonl'))
    original = Config.fromfile(root / base, lazy_import=False).to_dict()
    remote = Config.fromfile(
        root / f'pi05_paligemma_robolab_droid_{stem}_remote_inference.py',
        lazy_import=False).to_dict()
    contract = remote['themis']['server'].pop('continuous_sampling')
    assert contract['enabled'] is True
    assert contract['audit_path'] == str(tmp_path / 'audit.jsonl')
    assert remote == original
