# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import numpy as np
import torch

from .utils import (
    DeviceStr,
    get_params_grad,
    group_add,
    group_product,
    normalization,
    orthnormal,
)


def generic_generator(model, criterion, data, device: DeviceStr):
    """Calculates the per-sample gradient for most Pytorch models that implement `backward`.
    Default implementation used for PyHessian; the underlying code expects that this generator
    returns the size of the input and a pointer to the model at each step.
    Args:
        model (Any): The model to calculate per-sample gradients for.
        criterion (Any): Function that calculates the loss for the model.
        data (Any): Source of data for the model.
        device (DeviceStr): Device used for pyTorch calculations.

    Yields:
        The size of the current input (int) and the model.
    """
    for inputs, targets in data:
        model.zero_grad()  # clear gradients
        input_size = inputs.size(0)

        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)
        yield input_size, model


def dimenet_generator(model, criterion, data, device):
    """Example per-sample gradient generator for the Dimenet model architecture."""
    for batch in data:
        model.zero_grad()
        batch = batch.to(device)
        input_size = len(batch)

        # Compute loss using test_step which is consistent with how the model is used
        loss = model.test_step(batch, 0, None)
        loss.backward(create_graph=True)
        yield input_size, model


class PyHessian:
    def __init__(self, model, criterion, data, device, hessian_generator=generic_generator):
        self.model = model.eval()
        self.gen = hessian_generator
        self.params = [p for p in model.parameters()]
        self.criterion = criterion
        self.data = data
        self.device = device

    def dataloader_hv_product(self, v):
        THv = [torch.zeros(p.size()).to(self.device) for p in self.params]  # accumulate result
        num_data = 0
        for input_size, model in self.gen(self.model, self.criterion, self.data, self.device):
            params, gradsH = get_params_grad(model)
            model.zero_grad()
            Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, retain_graph=False)
            THv = [THv1 + Hv1 * float(input_size) + 0.0 for THv1, Hv1 in zip(THv, Hv, strict=False)]
            num_data += float(input_size)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """
        assert top_n >= 1
        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
            v = normalization(v)  # normalize the vector

            for _ in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        device = self.device
        trace_vhv = []
        trace = 0.0

        for _ in range(maxIter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            _, Hv = self.dataloader_hv_product(v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """
        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for _ in range(n_v):
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
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
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    _, w_prime = self.dataloader_hv_product(v)
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
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    _, w_prime = self.dataloader_hv_product(v)
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
