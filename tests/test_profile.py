import pytest
import pytest_html

from landscaper.topology_profile import generate_profile
from landscaper.plots import topology_profile


def test_generate_profile(landscape_2d, extras):
    mt = landscape_2d.get_sublevel_tree()
    profile = generate_profile(mt)
    svg = topology_profile(profile)

    extras.append(pytest_html.extras.png(svg.rasterize().as_data_uri()))
