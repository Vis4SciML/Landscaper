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

import torch
import math
import copy
from torch.autograd import Variable
import numpy as np

from pyhessian.utils import (
    group_product,
    group_add,
    normalization,
    get_params_grad,
    hessian_vector_product,
    orthnormal,
)


class PyHessian:
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        assert (data is not None and dataloader is None) or (
            data is None and dataloader is not None
        )
        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def dataloader_hv_product(self, v):
        assert self.params is not None and self.gradsH is not None

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [
            torch.zeros(p.size()).to(device) for p in self.params
        ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0.0 for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

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
        assert self.params is not None and self.gradsH is not None

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [
                torch.randn(p.size()).to(device) for p in self.params
            ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
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
        assert self.params is not None and self.gradsH is not None

        device = self.device
        trace_vhv = []
        trace = 0.0

        for i in range(maxIter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
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
        assert self.params is not None and self.gradsH is not None

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
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
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
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
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
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


class hessian(PyHessian):
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        super().__init__(model, criterion, data, dataloader, cuda)

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == "cuda":
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation


class hessian_pinn(PyHessian):
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(
        self, whole_model, model, criterion, data=None, dataloader=None, cuda=True
    ):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        super().__init__(model, criterion, data, dataloader, cuda)
        self.whole_model = whole_model

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == "cuda":
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            if torch.is_grad_enabled():
                self.whole_model.optimizer.zero_grad()
            # u_pred = self.whole_model.net_u(self.whole_model.x_u, self.whole_model.t_u)
            u_pred = self.model(torch.cat([self.inputs, self.targets], dim=1))
            u_pred_lb = self.whole_model.net_u(
                self.whole_model.x_bc_lb, self.whole_model.t_bc_lb
            )
            u_pred_ub = self.whole_model.net_u(
                self.whole_model.x_bc_ub, self.whole_model.t_bc_ub
            )
            if self.whole_model.nu != 0:
                u_pred_lb_x, u_pred_ub_x = self.whole_model.net_b_derivatives(
                    u_pred_lb,
                    u_pred_ub,
                    self.whole_model.x_bc_lb,
                    self.whole_model.x_bc_ub,
                )
            f_pred = self.whole_model.net_f(self.whole_model.x_f, self.whole_model.t_f)

            if self.whole_model.loss_style == "mean":
                # loss_u = torch.mean((self.whole_model.u - u_pred) ** 2)
                loss_u = torch.mean((self.targets - u_pred) ** 2)
                loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
                if self.whole_model.nu != 0:
                    loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
                loss_f = torch.mean(f_pred**2)
            elif self.whole_model.loss_style == "sum":
                # loss_u = torch.mean((self.targets - u_pred) ** 2)
                loss_u = torch.mean((self.targets - u_pred) ** 2)
                loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
                if self.whole_model.nu != 0:
                    loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
                loss_f = torch.sum(f_pred**2)

            loss = loss_u + loss_b + self.whole_model.L * loss_f

            print("loss: ", loss.item())
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation


class hessian_dimenet(PyHessian):
    """
    The class used to compute Hessian-related properties for the DimeNet/GeoGNN model:
        i) the top eigenvalues of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density

    This class is specifically designed to work with the DimeNet model structure,
    which uses the GeoGNN model and processes geometric graph data.
    """

    def __init__(self, model, criterion=None, data=None, dataloader=None, cuda=True):
        """
        model: the GeoGNN model that needs Hessian information
        criterion: the loss function (if None, uses model's test_step method)
        data: a single batch of data from the GeoGNN dataloader
        dataloader: the data loader including multiple batches of data
        cuda: whether to use GPU acceleration
        """

        super().__init__(model, criterion, data, dataloader, cuda)

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.batch = self.data
            if self.device == "cuda" and not self.batch.is_cuda:
                self.batch = self.batch.to("cuda")

            # If we only compute the Hessian information for a single batch, we can re-use the gradients
            self.model.zero_grad()

            # Use the model's test_step method which is designed for the DimeNet model
            # This internally handles the forward pass and loss computation
            loss = self.model.test_step(self.batch, 0, None)
            loss.backward(create_graph=True)

        # Extract parameters and gradients from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):
        """
        Compute the product of the Hessian and a vector v using the dataloader.
        This method computes Hv for each batch and averages the results.
        """
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [
            torch.zeros(p.size()).to(device) for p in self.params
        ]  # accumulate result

        # Handle dataloader iteration differently based on the data type
        if self.full_dataset:
            dataloader = self.data
            # When using a dataloader, iterate through it
            for batch in dataloader:
                self.model.zero_grad()
                batch = batch.to(device)
                tmp_num_data = len(batch)  # Get batch size

                # Compute loss using test_step which is consistent with how the model is used
                loss = self.model.test_step(batch, 0, None)
                loss.backward(create_graph=True)

                params, gradsH = get_params_grad(self.model)
                self.model.zero_grad()

                # Compute Hessian-vector product
                Hv = torch.autograd.grad(
                    gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
                )

                THv = [
                    THv1 + Hv1 * float(tmp_num_data) + 0.0 for THv1, Hv1 in zip(THv, Hv)
                ]
                num_data += float(tmp_num_data)
        else:
            # When using a single batch
            batch = self.data
            self.model.zero_grad()
            batch = batch.to(device)
            tmp_num_data = len(batch)  # Get batch size

            # Compute loss using test_step
            loss = self.model.test_step(batch, 0, None)
            loss.backward(create_graph=True)

            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()

            # Compute Hessian-vector product
            Hv = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )

            THv = Hv
            num_data = float(tmp_num_data)

        # Scale the result by the number of data points if we processed multiple batches
        if self.full_dataset:
            THv = [THv1 / float(num_data) for THv1 in THv]

        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
