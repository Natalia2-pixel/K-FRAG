import torch
from kfrag_cvpr.model import KFRAGRegionalRefinerV2

def test_refiner_shapes():
    model=KFRAGRegionalRefinerV2()
    images=torch.zeros(2,3,64,64)
    raw=torch.zeros(2,4,4,20)
    out=model(images,raw)
    assert out["refined_cell_logits20"].shape==(2,4,4,20)
    assert out["fragment_logits20"].shape==(2,20)
    assert out["cell_weights"].shape==(2,4,4)
