import pytest
import pytest_html
from utils import mpl_fig_to_report, svg_to_str

from landscaper import LossLandscape


@pytest.mark.slow
def test_compute(resnet_50, cifar10_test, torch_device, hessian_eigenvecs, resnet_criterion, extras):
    def loss_function(model, data):
        batch_loss = 0
        for d in data:
            tt, lbl_t = d
            output = model.forward(tt)
            loss = resnet_criterion(output, lbl_t)
            batch_loss += loss
        return batch_loss

    evals, evecs = hessian_eigenvecs
    print(evals)

    ls = LossLandscape.compute(
        resnet_50,
        cifar10_test,
        evecs,
        loss_function,
        dim=2,
        device=torch_device,
    )

    # draw some plots for visual inspection
    svg = ls.show_profile()
    extras.append(pytest_html.extras.svg(svg_to_str(svg)))

    ls.save("resnet50.npz")

    surf = ls.show(show=False)
    mpl_fig_to_report(surf, extras)

    ctr = ls.show_contour(show=False)
    mpl_fig_to_report(ctr, extras)
