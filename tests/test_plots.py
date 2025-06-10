import pytest
from utils import mpl_fig_to_report
import landscaper.plots as lsplt


def test_surface(landscape_2d, extras):
    f = landscape_2d.show(show=False)
    mpl_fig_to_report(f, extras)


def test_contour(landscape_2d, extras):
    f = landscape_2d.show_contour(show=False)
    mpl_fig_to_report(f, extras)


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
