import pytest
import pytest_html
from utils import svg_to_str

from landscaper.plots import topology_profile
from landscaper.topology_profile import generate_profile


@pytest.fixture
def profile_2d(rosenbrock_2d):
    mt = rosenbrock_2d.get_sublevel_tree()
    return generate_profile(mt)


@pytest.fixture
def profile_5d(rosenbrock_5d):
    mt = rosenbrock_5d.get_sublevel_tree()
    return generate_profile(mt)


@pytest.fixture
def profile(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("profile", ["profile_2d", "profile_5d"], indirect=True)
def test_generate_profile_grad(profile, extras):
    svg = topology_profile(profile, gradient=True)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))


def test_generate_profile_no_grad(profile_2d, extras):
    svg = topology_profile(profile_2d, gradient=False, y_axis=None)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))


def test_generate_profile_axis(profile_2d, extras):
    svg = topology_profile(profile_2d, gradient=False)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))
