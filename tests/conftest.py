import numpy as np
import pytest

from landscaper.landscape import LossLandscape


@pytest.fixture
def landscape_2d():
    # reset random seed
    rng = np.random.default_rng(123456)

    ranges = [np.linspace(-1, 1, 10) for x in range(2)]
    loss = rng.random((50, 2))

    return LossLandscape(loss, ranges)
