import pytest


@pytest.fixture
def landscape(request):
    return request.getfixturevalue(request.param)


@pytest.mark.slow
@pytest.mark.parametrize("landscape", ["rosenbrock_2d", "rosenbrock_5d"], indirect=True)
def test_smad_weighted_and_normalized(landscape):
    val = landscape.smad(normalize=True, weighted=True)
    print(val)


@pytest.mark.slow
@pytest.mark.parametrize("landscape", ["rosenbrock_2d", "rosenbrock_5d"], indirect=True)
def test_smad_unweighted_normalized(landscape):
    val = landscape.smad(normalize=True, weighted=False)
    print(val)
