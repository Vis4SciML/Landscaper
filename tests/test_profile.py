import pytest
import pytest_html

from landscaper.plots import topology_profile
from landscaper.topology_profile import generate_profile
import base64


@pytest.fixture
def profile(landscape_2d):
    mt = landscape_2d.get_sublevel_tree()
    return generate_profile(mt)


def test_generate_profile_grad(profile, extras):
    svg = topology_profile(profile, gradient=True)

    extras.append(
        pytest_html.extras.svg(
            f"data:image/svg+xml;base64,{base64.b64encode(svg.as_svg().encode('utf-8')).decode('utf-8')}"
        )
    )


def test_generate_profile_no_grad(profile, extras):
    svg = topology_profile(profile, gradient=False)

    extras.append(pytest_html.extras.png(svg.rasterize().as_data_uri()))
