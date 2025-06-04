import base64

import pytest
import pytest_html
import torch


def test_hvp(hessian_comp):
    # TODO: test against default implementation
    v = [torch.randn(p.size()) for p in hessian_comp.params]
    hessian_comp.hv_product(v)


def test_trace(hessian_comp):
    hessian_comp.trace()
