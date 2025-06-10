import pytest
import torch


@pytest.mark.dependency()
@pytest.mark.slow
def test_hvp(hessian_comp, torch_device):
    # TODO: test against default implementation
    v = [torch.randn(p.size(), device=torch_device) for p in hessian_comp.params]
    hessian_comp.hv_product(v)


@pytest.mark.dependency(depends=["test_hvp"])
@pytest.mark.slow
def test_trace(hessian_comp):
    hessian_comp.trace()
