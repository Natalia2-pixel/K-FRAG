"""Static release configuration/checkpoint contract; no model inference."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'config/KFRAG_V415_FROZEN_CONFIG.json').read_text())
assert (cfg['canonical_image_size'],cfg['regions'],cfg['region_size'],cfg['fragment_bits']) == (256,16,64,20)
assert cfg['protocol']['V4_N']==16 and cfg['protocol']['V4_K']==6
assert cfg['search']['COARSE_STRIDE']==2
assert cfg['search']['TOP_COARSE']==60
assert cfg['search']['REFINE_RADIUS']==10
assert cfg['search']['PHASE_TOP_K']==5
assert cfg['search']['PHASE_STEP']==16
assert cfg['final_recovery_function']=='blind_v412_phase_recovery'
assert cfg['rescue_wrapper_in_final_method'] is False
expected={
'KFRAG_V415_FINAL_MODELS.pt':'1cdfb9f0b1fbabd7272c67bfff73faa00b41c98db84d3443781ab1047a7886cf',
'fusion_model_state_dict.pt':'d3e6423532a7bae86a0748ddb0811123a9097e9e2894ae674ed2de49295030e1',
'v2_model_audit_state_dict.pt':'f5367ad822fb613c790fe9b055a76e992b8c133e5c6d8dc07f76c0ea1bf30937',
}
for name,digest in expected.items():
    path=ROOT/'checkpoints'/name
    assert path.is_file(), f'Missing {path}'
    actual=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    assert actual==digest, f'Checkpoint hash mismatch: {name}'
    print('CHECKPOINT SHA256 PASS:',name)
print('RELEASE CONTRACT CHECK: PASS (no neural inference executed)')
