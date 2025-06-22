"""
PACMANN: Point Adaptive Collocation Method for Artificial Neural Networks

Copyright 2025 C. Visser, A. Heinlein, B. Giovanardi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import math
import os
import random
import time

import numpy as np

import deepxde as dde


def PINN_Beltrami(train_number, 
                  random_seed_input               = True,
                  amount_collocation_points_input = 200,
                  learning_rate_input             = 1e-3,
                  save_collocation_points_input   = False,
                  collocation_type_input          = "Adam",
                  resample_period_input           = 50,
                  stepsize_input                  = 1e-4,
                  iteration_amount_input          = 5,
                  seed_input                      = 0000
                  ):

    #################### Input Parameters ####################

    a  = 1
    d  = 1
    Re = 100000

    if random_seed_input:
        random_seed = random.randint(0, 10000)
    else:
        random_seed = seed_input

    amount_collocation_points = amount_collocation_points_input
    learning_rate             = learning_rate_input
    save_collocation_points   = save_collocation_points_input
    collocation_type          = collocation_type_input
    resample_period           = resample_period_input
    stepsize                  = stepsize_input
    iteration_amount          = iteration_amount_input

    #########################################################

    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    dde.config.set_random_seed(random_seed)
    dde.config.set_default_float("float64")

    def pde(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x  = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y  = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z  = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t  = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x  = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y  = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z  = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t  = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x  = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y  = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z  = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t  = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )
        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]
    
    def pde_residual(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x  = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y  = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z  = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t  = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x  = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y  = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z  = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t  = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x  = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y  = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z  = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t  = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )

        r = momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2

        return r

    def residual_gradient_operator(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x  = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y  = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z  = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t  = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x  = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y  = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z  = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t  = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x  = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y  = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z  = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t  = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )

        r = momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2

        dr_X = dde.grad.jacobian(r, x, i=0)

        return dr_X

    def u_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def v_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def w_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def p_func(x):
        return (
            -0.5
            * a ** 2
            * (
                np.exp(2 * a * x[:, 0:1])
                + np.exp(2 * a * x[:, 1:2])
                + np.exp(2 * a * x[:, 2:3])
                + 2
                * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
                * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
                + 2
                * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
                * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
                + 2
                * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
                * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
            )
            * np.exp(-2 * d ** 2 * x[:, 3:4])
        )

    spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
    temporal_domain = dde.geometry.TimeDomain(0, 1)
    spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    boundary_condition_u = dde.icbc.DirichletBC(
        spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
    )
    boundary_condition_v = dde.icbc.DirichletBC(
        spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
    )
    boundary_condition_w = dde.icbc.DirichletBC(
        spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
    )

    initial_condition_u = dde.icbc.IC(
        spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
    )
    initial_condition_v = dde.icbc.IC(
        spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
    )
    initial_condition_w = dde.icbc.IC(
        spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
    )

    if collocation_type == "Uniform":
        data = dde.data.TimePDE(
            spatio_temporal_domain,
            pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ],
            num_domain=amount_collocation_points,
            num_boundary=2000,
            num_initial=500,
            num_test=10000,
            train_distribution="uniform",
        )
    elif collocation_type == "RAR" or collocation_type == "RAR_D":
        data = dde.data.TimePDE(
            spatio_temporal_domain,
            pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ],
            num_domain=amount_collocation_points // 2,
            num_boundary=2000,
            num_initial=500,
            num_test=10000,
        )
    else:
        data = dde.data.TimePDE(
            spatio_temporal_domain,
            pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ],
            num_domain=amount_collocation_points,
            num_boundary=2000,
            num_initial=500,
            num_test=10000,
        )

    net = dde.nn.FNN([4] + 4 * [64] + [4], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    if save_collocation_points:
        with open('resampled_points.txt', 'w') as f:
            """
            To obtain the collocation points, use train_x instead of train_x_all since only train_x is ordered, with BC points followed by the collocation points.
            By skipping over these BC points, we find the collocation points. Since BC points might be duplicated, one needs to skip over a multiple of the total number 
            of BC points to obtain the collocation points. https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.pde.PDE.train_x
            """
            f.write(f'0:\n')
            np.savetxt(f, model.data.train_x[10000:, :])

    #################### Sampling Methods ####################
    class Static(dde.callbacks.Callback):

        def __init__(self, period=50):
            super().__init__()
            self.period                     = period
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.points_added               = 0
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[10000:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False

    class Random_resampling(dde.callbacks.Callback):

        def __init__(self, N_col_points, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.save_points                = save_points
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            random_sample = spatio_temporal_domain.random_points(self.N_col_points)

            self.model.data.replace_with_anchors(random_sample)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

    class RAR(dde.callbacks.Callback):

        def __init__(self, N_col_points, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.save_points                = save_points
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.points_added               = 0
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.points_added >= amount_collocation_points / 2:
                return

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[10000:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False

            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)      

            X = spatio_temporal_domain.random_points(100000)
            f = model.predict(X, operator=pde_residual)
            err_eq = np.absolute(f)
            x_ids = np.argsort(err_eq, axis=0)

            X_sorted = np.take_along_axis(X, x_ids, axis=0)
            X_anchors = X_sorted[-points_added_per_resample:]

            data.add_anchors(X_anchors)

            self.points_added += points_added_per_resample

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

    class RAD(dde.callbacks.Callback):

        def __init__(self, N_col_points, save_points=False, c=1, k=1, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.save_points                = save_points
            self.c                          = c
            self.k                          = k
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0
            
            X = spatio_temporal_domain.random_points(100000)
            Y = np.abs(self.model.predict(X, operator=pde_residual)).astype(np.float64)
            err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(X), size=self.N_col_points, replace=False, p=err_eq_normalized)
            X_selected = X[X_ids]

            data.replace_with_anchors(X_selected)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

    class RAR_D(dde.callbacks.Callback):

        def __init__(self, N_col_points, c=2, k=0, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.save_points                = save_points
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.c                          = c
            self.k                          = k
            self.points_added               = 0
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.points_added >= amount_collocation_points / 2:
                return

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[10000:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)

            X = spatio_temporal_domain.random_points(100000)
            Y = np.abs(model.predict(X, operator=pde_residual)).astype(np.float64)
            err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(X), size=points_added_per_resample, replace=False, p=err_eq_normalized)

            data.add_anchors(X[X_ids])

            self.points_added += points_added_per_resample

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

    class Gradient_ascent(dde.callbacks.Callback):

        def __init__(self, N_col_points, stepsize, n_iterations, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.save_points                = save_points
            self.stepsize                   = stepsize
            self.iteration_amount           = n_iterations
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            for n in range(self.iteration_amount):
                residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)
                gradient_collocation_points = collocation_points + self.stepsize * residual_gradient
                collocation_points = self.replace_points(gradient_collocation_points)
            
            data.replace_with_anchors(collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points
    
    class Nonlinear_gradient_ascent(dde.callbacks.Callback):

        def __init__(self, N_col_points, stepsize, n_iterations, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.save_points                = save_points
            self.stepsize                   = stepsize
            self.first_resample             = True
            self.iteration_amount           = n_iterations

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            for n in range(self.iteration_amount):
                residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)
                gradient_collocation_points = collocation_points + self.stepsize * np.tanh(residual_gradient)
                collocation_points = self.replace_points(gradient_collocation_points)
            
            data.replace_with_anchors(collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points

    class RMSprop(dde.callbacks.Callback):

        def __init__(self, n_iterations, N_col_points, stepsize, beta = 0.999, epsilon = 10e-8, save_points=False, period=50):
            super().__init__()
            self.period = period
            self.N_col_points = N_col_points
            self.iteration_amount = n_iterations
            self.beta = beta
            self.epsilon = epsilon
            self.epochs_since_last_resample = 0
            self.total_resample_calls = 0
            self.save_points = save_points
            self.stepsize = stepsize
            self.first_resample = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            SdX = np.zeros((self.N_col_points, 2))

            for n in range(self.iteration_amount):
                residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)

                SdX = self.beta * SdX + (1 - self.beta) * residual_gradient ** 2
                collocation_points = collocation_points + self.stepsize * residual_gradient / (np.sqrt(SdX) + self.epsilon)

            new_collocation_points = self.replace_points(collocation_points)

            data.replace_with_anchors(new_collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points

    class Momentum(dde.callbacks.Callback):

        def __init__(self, n_iterations, N_col_points, stepsize, beta = 0.9, epsilon = 10e-8, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.iteration_amount           = n_iterations
            self.beta                       = beta
            self.epsilon                    = epsilon
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.save_points                = save_points
            self.stepsize                   = stepsize
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 2))

            for n in range(self.iteration_amount):
                residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)

                VdX = self.beta * VdX + (1 - self.beta) * residual_gradient

                collocation_points = collocation_points + self.stepsize * VdX

            new_collocation_points = self.replace_points(collocation_points)

            data.replace_with_anchors(new_collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points

    class Adam(dde.callbacks.Callback):

        def __init__(self, n_iterations, N_col_points, stepsize, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.iteration_amount           = n_iterations
            self.beta1                      = beta1
            self.beta2                      = beta2
            self.epsilon                    = epsilon
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.save_points                = save_points
            self.stepsize                   = stepsize
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 4))
            SdX = np.zeros((self.N_col_points, 4))

            for n in range(self.iteration_amount):
                residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)

                VdX = self.beta1 * VdX + (1 - self.beta1) * residual_gradient
                SdX = self.beta2 * SdX + (1 - self.beta2) * residual_gradient ** 2

                VdX_corrected = VdX / (1 - self.beta1 ** (n + 1))
                SdX_corrected = SdX / (1 - self.beta2 ** (n + 1))

                collocation_points = collocation_points + self.stepsize * VdX_corrected / (np.sqrt(SdX_corrected) + self.epsilon)

            new_collocation_points = self.replace_points(collocation_points)

            data.replace_with_anchors(new_collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points

    class Golden_section_search(dde.callbacks.Callback):

        def __init__(self, n_iterations, N_col_points, stepsize, save_points=False, period=50):
            super().__init__()
            self.period                     = period
            self.N_col_points               = N_col_points
            self.iteration_amount           = n_iterations
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.save_points                = save_points
            self.stepsize                   = stepsize
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[10000:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            residual_gradient = self.model.predict(collocation_points, operator=residual_gradient_operator)

            gr = (1 + np.sqrt(5)) / 2
            alpha = 1 - 1 / gr
            beta = 1 / gr

            a_i = collocation_points
            b_i = collocation_points + self.stepsize * residual_gradient 

            shifted = None

            for n in range(self.iteration_amount):
                x_l = a_i + alpha * (b_i - a_i)
                x_r = a_i + beta * (b_i - a_i)

                if shifted is None:
                    f_l = self.model.predict(x_l, operator=pde_residual)
                    f_r = self.model.predict(x_r, operator=pde_residual)
                else:
                    f_l_prev = f_l
                    f_l = np.where(shifted == 0, f_r, self.model.predict(x_l, operator=pde_residual))
                    f_r = np.where(shifted == 1, f_l_prev, self.model.predict(x_r, operator=pde_residual))

                a_i = np.where(f_l > f_r, a_i, x_l)
                b_i = np.where(f_l > f_r, x_r, b_i)
                shifted = np.where(f_l > f_r, int(1), int(0))  
            
            iterated_points = (a_i + b_i) / 2

            new_collocation_points = self.replace_points(iterated_points)

            data.replace_with_anchors(new_collocation_points)

            if self.save_points:
                with open('resampled_points.txt', 'a') as f:
                    f.write(f'{self.total_resample_calls}\n')
                    np.savetxt(f, self.model.data.train_x_all)

        def replace_points(self, new_collocation_points):
            inside_space_ids = spatial_domain.inside(new_collocation_points[:,0:3])
            inside_time_ids = temporal_domain.inside(new_collocation_points[:,3])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = spatio_temporal_domain.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points
    
    # Non-adaptive
    if collocation_type == "Uniform":
        resampler = [Static(period=resample_period)]

    elif collocation_type == "Hammersley":
        resampler = [Static(period=resample_period)]

    elif collocation_type == "Random_resampling":
        resampler = [Random_resampling(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period)]

    # Adaptive
    elif collocation_type == "RAR":
        resampler = [RAR(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period)]

    elif collocation_type == "RAD":
        resampler = [RAD(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, c=1, k=1)]

    elif collocation_type == "RAR_D":
        resampler = [RAR_D(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, c=2, k=0)]

    # PACMANN
    elif collocation_type == "Gradient_ascent":
        resampler = [Gradient_ascent(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    elif collocation_type == "Nonlinear_gradient_ascent":
        resampler = [Nonlinear_gradient_ascent(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    elif collocation_type == "RMSprop":
        resampler = [RMSprop(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    elif collocation_type == "Momentum":
        resampler = [Momentum(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    elif collocation_type == "Adam":
        resampler = [Adam(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    elif collocation_type == "Golden_section_search":
        resampler = [Golden_section_search(N_col_points=amount_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=iteration_amount)]

    #########################################################


    start_time = time.time()

    for n in range(4):
        model.compile("adam", lr=learning_rate)
        model.train(iterations=7000, display_every=100, callbacks=resampler)

        model.compile("L-BFGS")
        model.train(display_every=100)

    model.compile("adam", lr=learning_rate)
    model.train(iterations=7000, display_every=100, callbacks=resampler)

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)

    train_time = time.time() - start_time

    x, y, z, t = np.meshgrid(
        np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(0, 1, 5)
    )

    X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T

    output = model.predict(X)

    u_pred = output[:, 0].reshape(-1)
    v_pred = output[:, 1].reshape(-1)
    w_pred = output[:, 2].reshape(-1)
    p_pred = output[:, 3].reshape(-1)

    u_exact = u_func(X).reshape(-1)
    v_exact = v_func(X).reshape(-1)
    w_exact = w_func(X).reshape(-1)
    p_exact = p_func(X).reshape(-1)

    f = model.predict(X, operator=pde)

    l2_difference_u_1 = dde.metrics.l2_relative_error(u_exact, u_pred)
    l2_difference_v_1 = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_w_1 = dde.metrics.l2_relative_error(w_exact, w_pred)
    l2_difference_p_1 = dde.metrics.l2_relative_error(p_exact, p_pred)
    residual_1 = np.mean(np.absolute(f))

    print("Accuracy")
    print("Mean residual:", residual_1)
    print("L2 relative error in u:", l2_difference_u_1)
    print("L2 relative error in v:", l2_difference_v_1)
    print("L2 relative error in w:", l2_difference_w_1)

    if not os.path.exists(str(train_number)):
        os.makedirs(str(train_number))

    output = [collocation_type, train_time, l2_difference_u_1, l2_difference_v_1, l2_difference_w_1, l2_difference_p_1, learning_rate, stepsize, amount_collocation_points, random_seed, resample_period, iteration_amount]
    output_file_path = os.path.join(str(train_number), 'output.csv')

    with open(output_file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(output)

if __name__ == "__main__":
    PINN_Beltrami(train_number=0)