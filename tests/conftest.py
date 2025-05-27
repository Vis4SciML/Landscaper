import pytest
import numpy as np
from landscaper.landscape import LossLandscape

rng = np.random.default_rng(123456)


@pytest.fixture
def landscape_2d():
    ranges = [np.linspace(-1, 1, 10) for x in range(2)]
    loss = rng.random((50, 2))

    return LossLandscape(loss, ranges)
