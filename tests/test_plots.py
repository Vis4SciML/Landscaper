import pytest
from utils import mpl_fig_to_report

import landscaper.plots as lsplt


def test_surface(landscape_2d, extras):
    f = landscape_2d.show(show=False)
    mpl_fig_to_report(f, extras)


def test_contour(landscape_2d, extras):
    f = landscape_2d.show_contour(show=False)
    mpl_fig_to_report(f, extras)


def test_contour_with_vmin_vmax(landscape_2d, extras):
    """Test contour plot with custom vmin and vmax parameters."""
    f = landscape_2d.show_contour(show=False, vmin=0.1, vmax=0.9)
    mpl_fig_to_report(f, extras)


def test_surface_with_vmin_vmax(landscape_2d, extras):
    """Test 3D surface plot with custom vmin and vmax parameters."""
    f = landscape_2d.show(show=False, vmin=0.1, vmax=0.9)
    mpl_fig_to_report(f, extras)


def test_surface_with_azim(landscape_2d, extras):
    """Test 3D surface plot with custom azimuthal angle (z-axis rotation)."""
    # Test with different rotation angles
    f1 = landscape_2d.show(show=False, azim=0)
    mpl_fig_to_report(f1, extras)
    
    f2 = landscape_2d.show(show=False, azim=90)
    mpl_fig_to_report(f2, extras)
    
    f3 = landscape_2d.show(show=False, azim=180)
    mpl_fig_to_report(f3, extras)


def test_persistence_barcode(landscape_2d, extras):
    f = landscape_2d.show_persistence_barcode(show=False)
    mpl_fig_to_report(f, extras)

@pytest.mark.slow
def test_hessian_density_plt(hessian_density, extras):
    eigen, weight = hessian_density
    f = lsplt.hessian_density(eigen, weight, show=False)
    mpl_fig_to_report(f, extras)

@pytest.mark.slow
def test_hessian_eigen_plt(hessian_eigenvecs, extras):
    evals, evecs = hessian_eigenvecs
    f = lsplt.hessian_eigenvalues(evals, show=False)
    mpl_fig_to_report(f, extras)
