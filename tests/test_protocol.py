import pytest
pytest.importorskip("reedsolo")
from kfrag_cvpr.protocol import v3_issue, fragments_to_bits20

def test_protocol_shape():
    fragments=v3_issue(b"12345678", b"demo-test-key")
    bits=fragments_to_bits20(fragments)
    assert len(fragments)==16
    assert len(bits)==16
    assert all(len(row)==20 for row in bits)
