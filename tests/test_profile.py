import pytest
import pytest_html

from landscaper.plots import topology_profile
from landscaper.topology_profile import generate_profile
from utils import svg_to_str


@pytest.fixture
def profile(landscape_2d):
    mt = landscape_2d.get_sublevel_tree()
    return generate_profile(mt)


def test_generate_profile_grad(profile, extras):
    svg = topology_profile(profile, gradient=True)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))


def test_generate_profile_no_grad(profile, extras):
    svg = topology_profile(profile, gradient=False, y_axis=None)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))


def test_generate_profile_axis(profile, extras):
    svg = topology_profile(profile, gradient=False)
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))
