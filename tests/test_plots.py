import pytest
from utils import mpl_fig_to_report

import landscaper.plots as lsplt


def test_surface(rosenbrock_2d, extras):
    f = rosenbrock_2d.show(show=False)
    mpl_fig_to_report(f, extras)


def test_contour(rosenbrock_2d, extras):
    f = rosenbrock_2d.show_contour(show=False)
    mpl_fig_to_report(f, extras)


def test_contour_with_vmin_vmax(rosenbrock_2d, extras):
    """Test contour plot with custom vmin and vmax parameters."""
    f = rosenbrock_2d.show_contour(show=False, vmin=0.1, vmax=0.9)
    mpl_fig_to_report(f, extras)


def test_surface_with_vmin_vmax(rosenbrock_2d, extras):
    """Test 3D surface plot with custom vmin and vmax parameters."""
    f = rosenbrock_2d.show(show=False, vmin=0.1, vmax=0.9)
    mpl_fig_to_report(f, extras)


def test_persistence_barcode(rosenbrock_2d, extras):
    f = rosenbrock_2d.show_persistence_barcode(show=False)
    mpl_fig_to_report(f, extras)


def test_sublevel_tree(rosenbrock_2d, extras):
    f = rosenbrock_2d.show_sublevel_tree(show=False, log_scale=True)
    mpl_fig_to_report(f, extras)


def test_super_tree(rosenbrock_2d, extras):
    f = rosenbrock_2d.show_super_tree(show=False)
    mpl_fig_to_report(f, extras)


@pytest.mark.slow
def test_hessian_density_plt(hessian_comp, extras):
    eigen, weight = hessian_comp.density()
    f = lsplt.hessian_density(eigen, weight, show=False)
    mpl_fig_to_report(f, extras)


@pytest.mark.slow
def test_hessian_eigen_plt(hessian_eigenvecs, extras):
    evals, evecs = hessian_eigenvecs
    f = lsplt.hessian_eigenvalues(evals, show=False)
    mpl_fig_to_report(f, extras)
