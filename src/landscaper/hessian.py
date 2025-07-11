"""Module for calculating the hessian.

This code was copied and modified from PyHessian (https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py).
"""

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    DeviceStr,
    group_add,
    group_product,
    normalization,
    orthnormal,
)


def generic_generator(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data: Any,
    device: DeviceStr,
) -> Generator[tuple[int, torch.Tensor], None, None]:
    """Calculates the per-sample gradient for the model.

    Default implementation used for PyHessian; the underlying code expects that this generator
    returns the size of the input and the gradient tensor at each step.

    Args:
        model (torch.nn.Module): The model to calculate per-sample gradients for.
        criterion (torch.nn.Module): Function that calculates the loss for the model.
        data (Any): Source of data for the model.
        device (DeviceStr): Device used for pyTorch calculations.

    Yields:
        The size of the current input (int) and the gradient for that sample.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    for sample, target in data:
        outputs = model.forward(sample)
        loss = criterion(outputs, target)

        # instead of loss.backward we directly compute the gradient to avoid overwriting the gradient in place
        grads = torch.autograd.grad(
            loss, params, create_graph=True, materialize_grads=True
        )
        yield sample.size(0), grads


"""
def generic_generator_reverse_over_forward(
    model: torch.nn.Module, criterion: torch.nn.Module, data: Any, device: DeviceStr, v
) -> Generator[tuple[int, torch.Tensor], None, None]:
    grads = []
    for (inputs, targets), vv in zip(data, v):
        input_size = inputs.size(0)
        jvp_func = lambda x: torch.func.jvp(
            lambda xx: criterion(model.forward(xx), targets),
            (x,),
            (vv,),
        )[1]
        grads.append(torch.func.grad(jvp_func)(inputs))

    yield input_size, grads
"""


def _is_model_complex(model):
    for p in model.parameters():
        if torch.is_complex(p):
            return True
    return False


def dimenet_generator(
    model: torch.nn.Module,
    criterion: torch.nn.Module | torch.Tensor,
    data: Any,
    device: DeviceStr,
) -> Generator[tuple[int, torch.Tensor], None, None]:
    """Calculates the per-sample gradient for DimeNet models.

    Args:
        model (torch.nn.Module): The DimeNet model to calculate per-sample gradients for.
        criterion (torch.nn.Module): Function that calculates the loss for the model.
        data (Any): Source of data for the model.
        device (DeviceStr): Device used for pyTorch calculations.

    Yields:
        The size of the current input (int) and the gradient.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    for batch in data:
        input_size = len(batch)

        # Move batch to the correct device
        batch = batch.to(device)

        # Compute loss using test_step which is consistent with how the model is used
        loss = model.test_step(batch, 0, None)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        yield input_size, grads


class PyHessian:
    """PyHessian class for computing Hessian-related quantities.

    This class provides methods to compute eigenvalues, eigenvectors, trace, and density of the Hessian
    matrix using various methods such as power iteration, Hutchinson's method, and stochastic Lanczos algorithm.
    It supports different model architectures and can be used with custom data loaders.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module | torch.Tensor,
        data: Any,
        device: DeviceStr,
        hessian_generator: Callable[
            [torch.nn.Module, torch.nn.Module | torch.Tensor, Any, DeviceStr, Any],
            Generator[tuple[int, torch.Tensor], None, None],
        ] = generic_generator,
        try_cache: bool = False,
        use_complex: bool = False,
    ):
        """Initializes the PyHessian class.

        Args:
            model (torch.nn.Module): The model for which the Hessian is computed.
            criterion (torch.nn.Module): The loss function used for training the model.
            data (torch.utils.data.DataLoader): DataLoader providing the training data.
            device (DeviceStr): Device to run the computations on (e.g., 'cpu' or 'cuda').
            hessian_generator (callable, optional): Function to generate per-sample gradients.
                Defaults to generic_generator.
            try_cache (bool): Defaults to false. Caches per-sample gradients along with their computational graphs.
                Should make the computation faster, but can cause out of memory errors. If you run into memory problems,
                try setting this to false first.
            use_complex (bool): Defaults to false. Forces the calculator to use complex values when performing
                computations. This is determined automatically, but this kwarg is included as a backup.
        """
        if model.training:
            print(
                "Setting model to eval mode. PyHessian will not work with models in training mode!"
            )
            self.model = model.eval()
        else:
            self.model = model

        self.gen = hessian_generator
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.criterion = criterion
        self.data = data
        self.device = device
        self.use_complex = _is_model_complex(self.model) or use_complex

        if self.use_complex:
            print(
                "Complex parameters detected in model. Results will be complex tensors."
            )

        if try_cache:
            grad_cache = []
            for input_size, grads in self.gen(
                self.model, self.criterion, self.data, self.device
            ):
                grad_cache.append((input_size, grads))
            self.grad_cache = grad_cache
            self.gen = lambda *args: (x for x in self.grad_cache)
        else:
            self.grad_cache = None

    def hv_product(self, v: list[torch.Tensor]) -> tuple[float, list[torch.Tensor]]:
        """Computes the product of the Hessian-vector product (Hv) for the data.

        Args:
            v (list[torch.Tensor]): A list of tensors representing the vector to multiply with the Hessian.

        Returns:
            tuple: A tuple containing the eigenvalue (float) and the Hessian-vector product (list of tensors).
        """
        THv = [torch.zeros_like(p) for p in self.params]  # accumulate result

        if self.use_complex:
            THv = [
                torch.complex(t, t.clone()) if not torch.is_complex(t) else t
                for t in THv
            ]

        num_data = 0
        for input_size, grads in self.gen(
            self.model, self.criterion, self.data, self.device
        ):
            if self.use_complex:
                grads = [
                    (
                        torch.complex(g, torch.zeros_like(g))
                        if not torch.is_complex(g)
                        else g
                    )
                    for g in grads
                ]

            Hv = torch.autograd.grad(
                grads,
                self.params,
                grad_outputs=v,
                retain_graph=self.grad_cache is not None,
            )

            THv = [
                THv1 + Hv1 * float(input_size) + 0.0
                for THv1, Hv1 in zip(THv, Hv, strict=False)
            ]
            num_data += float(input_size)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(
        self,
        maxIter: int = 100,
        tol: float = 1e-3,
        top_n: int = 1,
    ) -> tuple[list[float], list[list[torch.Tensor]]]:
        """Computes the top_n eigenvalues using power iteration method.

        Args:
            maxIter (int, optional): Maximum iterations used to compute each single eigenvalue. Defaults to 100.
            tol (float, optional): The relative tolerance between two consecutive eigenvalue computations
                from power iteration. Defaults to 1e-3.
            top_n (int, optional): The number of top eigenvalues to compute. Defaults to 1.

        Returns:
            tuple[list[float], list[list[torch.Tensor]]]: A tuple containing the eigenvalues and
                their corresponding eigenvectors.
        """
        assert top_n >= 1 and not self.model.training
        eigenvalues = []
        eigenvectors = []

        with tqdm(total=top_n, desc="Eigenvectors computed", leave=True) as pbar:
            computed_dim = 0

            while computed_dim < top_n:
                eigenvalue = None
                v = [torch.randn_like(p) for p in self.params]  # generate random vector

                if self.use_complex:
                    v = [
                        (
                            torch.complex(vv, torch.randn_like(vv))
                            if not torch.is_complex(vv)
                            else vv
                        )
                        for vv in v
                    ]

                v = normalization(v)  # normalize the vector

                ibar = tqdm(
                    range(maxIter),
                    total=maxIter,
                    desc="Iteration",
                    leave=False,
                    position=1,
                )
                for _ in ibar:
                    v = orthnormal(v, eigenvectors)

                    tmp_eigenvalue, Hv = self.hv_product(v)
                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        spec_gap = abs(eigenvalue - tmp_eigenvalue) / (
                            abs(eigenvalue) + 1e-6
                        )
                        ibar.set_description(f"Spectral gap: {spec_gap}")
                        if spec_gap < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                eigenvalues.append(eigenvalue)
                eigenvectors.append(v)
                computed_dim += 1

                pbar.update(1)

        return eigenvalues, eigenvectors

    def trace(self, maxIter: int = 100, tol: float = 1e-3) -> list[float]:
        """Computes the trace of the Hessian using Hutchinson's method.

        Args:
            maxIter (int): Maximum iterations used to compute the trace. Defaults to 100.
            tol (float): The relative tolerance for convergence. Defaults to 1e-3.

        Returns:
            list[float]: A list containing the trace of the Hessian computed over the iterations.
        """
        assert not self.model.training

        trace_vhv = []
        trace = 0.0

        for _ in range(maxIter):
            v = [torch.randint_like(p, high=2) for p in self.params]

            if self.use_complex:
                v = [
                    (
                        torch.complex(vv, torch.randint_like(vv, high=2))
                        if not torch.is_complex(vv)
                        else vv
                    )
                    for vv in v
                ]

            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            _, Hv = self.hv_product(v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(
        self, iter: int = 100, n_v: int = 1
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Computes the estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ).

        Args:
            iter (int): Number of iterations used to compute the trace. Defaults to 100.
            n_v (int): Number of SLQ runs. Defaults to 1.

        Returns:
            tuple[list[list[float]], list[list[float]]]: A tuple containing two lists:
                - eigen_list_full: List of eigenvalues from each SLQ run.
                - weight_list_full: List of weights corresponding to the eigenvalues.
        """
        assert not self.model.training

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for _ in range(n_v):
            v = [torch.randint_like(p, high=2) for p in self.params]

            if self.use_complex:
                v = [
                    (
                        torch.complex(vv, torch.randint_like(vv, high=2))
                        if not torch.is_complex(vv)
                        else vv
                    )
                    for vv in v
                ]

            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                w_prime = [torch.zeros_like(p) for p in self.params]

                if self.use_complex:
                    w_prime = [
                        (
                            torch.complex(vv, torch.zeros_like(vv))
                            if not torch.is_complex(vv)
                            else vv
                        )
                        for vv in w_prime
                    ]

                if i == 0:
                    _, w_prime = self.hv_product(v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.0:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn_like(p) for p in self.params]
                        if self.use_complex:
                            w = [
                                (
                                    torch.complex(vv, torch.randn_like(vv))
                                    if not torch.is_complex(vv)
                                    else vv
                                )
                                for vv in w
                            ]

                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    _, w_prime = self.hv_product(v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.linalg.eig(T)

            eigen_list = a_
            weight_list = torch.pow(b_, 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
