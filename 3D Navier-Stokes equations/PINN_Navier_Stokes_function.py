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
import torch

import deepxde as dde


def PINN_NS(train_number, 
            random_seed_input               = True,
            number_collocation_points_input = 250,
            learning_rate_input             = 1e-3,
            save_collocation_points_input   = False,
            collocation_type_input          = "Adam",
            resample_period_input           = 50,
            stepsize_input                  = 1e-4,
            number_of_iterations_input      = 5,
            seed_input                      = 0000
            ):

    #################### Input Parameters ####################

    Re = 1e3

    if random_seed_input:
        random_seed = random.randint(0, 10000)
    else:
        random_seed = seed_input

    number_collocation_points = number_collocation_points_input
    learning_rate             = learning_rate_input
    save_collocation_points   = save_collocation_points_input
    collocation_type          = collocation_type_input
    resample_period           = resample_period_input
    stepsize                  = stepsize_input
    number_of_iterations      = number_of_iterations_input

    #########################################################

    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    dde.config.set_random_seed(random_seed)
    dde.config.set_default_float("float64")

    def pde(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

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
            1 / Re * u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
            - ((Re*(-2*torch.sin(Y)**2 + torch.sin(Z)**2)*torch.sin(X) + 2*torch.exp(T)*torch.sin(Y)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(X)/Re)
        )
        momentum_y = (
            1 / Re * v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
            - ((Re*(-2*torch.sin(X)**2 + torch.sin(Z)**2)*torch.sin(Y) + 2*torch.exp(T)*torch.sin(X)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(Y)/Re)
        )
        momentum_z = (
            1 / Re * w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
            - (2*(Re*(torch.cos(X)**2 + torch.cos(Y)**2 - 2)*torch.sin(Z) - 2*torch.exp(T)*torch.sin(X)*torch.sin(Y))*torch.exp(-2*T)*torch.cos(Z)/Re)
        )

        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]
    
    def pde_residual(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

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
            1 / Re * u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
            - ((Re*(-2*torch.sin(Y)**2 + torch.sin(Z)**2)*torch.sin(X) + 2*torch.exp(T)*torch.sin(Y)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(X)/Re)
        )
        momentum_y = (
            1 / Re * v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
            - ((Re*(-2*torch.sin(X)**2 + torch.sin(Z)**2)*torch.sin(Y) + 2*torch.exp(T)*torch.sin(X)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(Y)/Re)
        )
        momentum_z = (
            1 / Re * w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
            - (2*(Re*(torch.cos(X)**2 + torch.cos(Y)**2 - 2)*torch.sin(Z) - 2*torch.exp(T)*torch.sin(X)*torch.sin(Y))*torch.exp(-2*T)*torch.cos(Z)/Re)
        )

        r = momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2

        return r

    def residual_gradient_operator(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

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
            1 / Re * u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
            - ((Re*(-2*torch.sin(Y)**2 + torch.sin(Z)**2)*torch.sin(X) + 2*torch.exp(T)*torch.sin(Y)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(X)/Re)
        )
        momentum_y = (
            1 / Re * v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
            - ((Re*(-2*torch.sin(X)**2 + torch.sin(Z)**2)*torch.sin(Y) + 2*torch.exp(T)*torch.sin(X)*torch.sin(Z))*torch.exp(-2*T)*torch.cos(Y)/Re)
        )
        momentum_z = (
            1 / Re * w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
            - (2*(Re*(torch.cos(X)**2 + torch.cos(Y)**2 - 2)*torch.sin(Z) - 2*torch.exp(T)*torch.sin(X)*torch.sin(Y))*torch.exp(-2*T)*torch.cos(Z)/Re)
        )

        r = momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2

        dr_X = dde.grad.jacobian(r, x, i=0)

        return dr_X

    def u_func(x):
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return (
            np.cos(X) * np.sin(Y) * np.sin(Z) * np.exp(-T)
        )


    def v_func(x):
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return (
            np.sin(X) * np.cos(Y) * np.sin(Z) * np.exp(-T)
        )


    def w_func(x):
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return (
            np.sin(X) * np.sin(Y) * np.cos(Z) * np.exp(-T) * -2
        )


    def p_func(x):
        X, Y, Z, T = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return (
            np.cos(X) * np.cos(Y) * np.cos(Z) * np.exp(-T)
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
    boundary_condition_p = dde.icbc.DirichletBC(
        spatio_temporal_domain, p_func, lambda _, on_boundary: on_boundary, component=3
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
    initial_condition_p = dde.icbc.IC(
        spatio_temporal_domain, p_func, lambda _, on_initial: on_initial, component=3
    )

    if collocation_type == "Uniform":
        def cube_points(axis, fixed_value, pts_number):
            x_bc = np.linspace(-1, 1, pts_number)
            y_bc = np.linspace(-1, 1, pts_number)
            z_bc = np.linspace(-1, 1, pts_number)

            if axis == 'x':
                x_bc = np.ones(2) * fixed_value
            elif axis == 'y':
                y_bc = np.ones(2) * fixed_value
            elif axis == 'z':
                z_bc = np.ones(2) * fixed_value
                

            X_BC, Y_BC, Z_BC = np.meshgrid(x_bc,
                                           y_bc,
                                           z_bc)
            
            pts = np.stack([X_BC.ravel(),
                            Y_BC.ravel(),
                            Z_BC.ravel()], axis=1)
            
            return pts

        t_cube_pts = np.linspace(0, 1, 3+1)[1:]

        bc_cube_pts = np.vstack([cube_points('x', 1, 8),
                                 cube_points('x', -1, 8),
                                 cube_points('y', 1, 8),
                                 cube_points('y', -1, 8),
                                 cube_points('z', 1, 8),
                                 cube_points('z', -1, 8)])

        bc_unique_cube_pts = np.unique(bc_cube_pts, axis=0)
        bc_unique_cube_pts_repeated = np.repeat(bc_unique_cube_pts, len(t_cube_pts), axis=0)

        t_cube_tiled = np.sort(np.tile(t_cube_pts, len(bc_unique_cube_pts)))[:, np.newaxis]

        bc_points = np.hstack([bc_unique_cube_pts_repeated, t_cube_tiled])

        ic_cube_pts = np.vstack([cube_points('x', 1, 7),
                                 cube_points('x', -1, 7),
                                 cube_points('y', 1, 7),
                                 cube_points('y', -1, 7),
                                 cube_points('z', 1, 7),
                                 cube_points('z', -1, 7)])

        ic_unique_cube_pts = np.unique(ic_cube_pts, axis=0)

        ic_points = np.hstack([ic_unique_cube_pts, np.zeros(len(ic_unique_cube_pts))[:, np.newaxis]])

        uniform_boundary_condition_u = dde.icbc.PointSetBC(
            bc_points, u_func(bc_points), 0
        )
        uniform_boundary_condition_v = dde.icbc.PointSetBC(
            bc_points, v_func(bc_points), 1
        )
        uniform_boundary_condition_w = dde.icbc.PointSetBC(
            bc_points, w_func(bc_points), 2
        )
        uniform_boundary_condition_p = dde.icbc.PointSetBC(
            bc_points, p_func(bc_points), 3
        )

        uniform_initial_condition_u = dde.icbc.PointSetBC(
            ic_points, u_func(ic_points), 0
        )
        uniform_initial_condition_v = dde.icbc.PointSetBC(
            ic_points, v_func(ic_points), 1
        )
        uniform_initial_condition_w = dde.icbc.PointSetBC(
            ic_points, w_func(ic_points), 2
        )
        uniform_initial_condition_p = dde.icbc.PointSetBC(
            ic_points, p_func(ic_points), 3
        )

        data = dde.data.TimePDE(
            spatio_temporal_domain,
            pde,
            [
                uniform_boundary_condition_u,
                uniform_boundary_condition_v,
                uniform_boundary_condition_w,
                uniform_boundary_condition_p,
                uniform_initial_condition_u,
                uniform_initial_condition_v,
                uniform_initial_condition_w,
                uniform_initial_condition_p,
            ],
            num_domain=number_collocation_points,
            num_boundary=0,
            num_initial=0,
            num_test=10000,
        )
    elif collocation_type == "RAR" or collocation_type == "RAR_D":
        data = dde.data.TimePDE(
            spatio_temporal_domain,
            pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                boundary_condition_p,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
                initial_condition_p,
            ],
            num_domain=number_collocation_points // 2,
            num_boundary=1000,
            num_initial=250,
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
                boundary_condition_p,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
                initial_condition_p,
            ],
            num_domain=number_collocation_points,
            num_boundary=1000,
            num_initial=250,
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
            np.savetxt(f, model.data.train_x[6250:, :])

    #################### Sampling Methods ####################
    class Static_uniform(dde.callbacks.Callback):

        def __init__(self, period=50):
            super().__init__()
            self.period                     = period
            self.epochs_since_last_resample = 0
            self.total_resample_calls       = 0
            self.points_added               = 0
            self.first_epoch                = True

        def on_epoch_begin(self):
            if self.first_epoch:
                x_uniform = np.linspace(-1, 1, 4)
                y_uniform = np.linspace(-1, 1, 4)
                z_uniform = np.linspace(-1, 1, 4)
                t_uniform = np.linspace(0, 1, 4)

                X_uniform, Y_uniform, Z_uniform, T_uniform = np.meshgrid(x_uniform,
                                                                         y_uniform,
                                                                         z_uniform,
                                                                         t_uniform)
                
                collocation_pts = np.stack([
                    X_uniform.ravel(), 
                    Y_uniform.ravel(), 
                    Z_uniform.ravel(), 
                    T_uniform.ravel()
                ], axis=1)

                data.replace_with_anchors(collocation_pts)
                self.first_epoch = False

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
                collocation_points_extracted = self.model.data.train_x[6250:, :]
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

            if self.points_added >= number_collocation_points / 2:
                return

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[6250:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False

            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((number_collocation_points // 2) / number_of_resamples)      

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

            if self.points_added >= number_collocation_points / 2:
                return

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[6250:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((number_collocation_points // 2) / number_of_resamples)

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
            self.number_of_iterations       = n_iterations
            self.first_resample             = True

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[6250:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            for n in range(self.number_of_iterations):
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
            self.number_of_iterations       = n_iterations

        def on_epoch_end(self):
            self.epochs_since_last_resample += 1
            self.total_resample_calls += 1
            if self.epochs_since_last_resample < self.period:
                return
            self.epochs_since_last_resample = 0

            if self.first_resample:
                collocation_points = self.model.data.train_x[6250:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            for n in range(self.number_of_iterations):
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
            self.number_of_iterations = n_iterations
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
                collocation_points = self.model.data.train_x[6250:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all

            SdX = np.zeros((self.N_col_points, 2))

            for n in range(self.number_of_iterations):
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
            self.number_of_iterations       = n_iterations
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
                collocation_points = self.model.data.train_x[6250:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 2))

            for n in range(self.number_of_iterations):
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
            self.number_of_iterations       = n_iterations
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
                collocation_points = self.model.data.train_x[6250:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 4))
            SdX = np.zeros((self.N_col_points, 4))

            for n in range(self.number_of_iterations):
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
            self.number_of_iterations       = n_iterations
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
                collocation_points = self.model.data.train_x[6250:, :]
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

            for n in range(self.number_of_iterations):
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
        resampler = [Static_uniform(period=resample_period)]

    elif collocation_type == "Hammersley":
        resampler = [Static(period=resample_period)]

    elif collocation_type == "Random_resampling":
        resampler = [Random_resampling(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period)]

    # Adaptive
    elif collocation_type == "RAR":
        resampler = [RAR(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period)]

    elif collocation_type == "RAD":
        resampler = [RAD(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, c=1, k=1)]

    elif collocation_type == "RAR_D":
        resampler = [RAR_D(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, c=2, k=0)]

    # PACMANN
    elif collocation_type == "Gradient_ascent":
        resampler = [Gradient_ascent(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    elif collocation_type == "Nonlinear_gradient_ascent":
        resampler = [Nonlinear_gradient_ascent(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    elif collocation_type == "RMSprop":
        resampler = [RMSprop(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    elif collocation_type == "Momentum":
        resampler = [Momentum(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    elif collocation_type == "Adam":
        resampler = [Adam(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    elif collocation_type == "Golden_section_search":
        resampler = [Golden_section_search(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    #########################################################

    class H1_norm_callback(dde.callbacks.Callback):

        def __init__(self, X_eval, train_number):
            super().__init__()
            self.X_eval = dde.backend.as_tensor(X_eval, dtype=dde.backend.float32)
            self.np_X_eval = X_eval

        def grad_solution(self, X_eval):
            x = X_eval[:,0:1]
            y = X_eval[:,1:2]
            z = X_eval[:,2:3]
            t = X_eval[:,3:4]

            u_x = -np.exp(-t)*np.sin(x)*np.sin(y)*np.sin(z)
            u_y = np.exp(-t)*np.sin(z)*np.cos(x)*np.cos(y)
            u_z = np.exp(-t)*np.sin(y)*np.cos(x)*np.cos(z)

            v_x = np.exp(-t)*np.sin(z)*np.cos(x)*np.cos(y)
            v_y = -np.exp(-t)*np.sin(x)*np.sin(y)*np.sin(z)
            v_z = np.exp(-t)*np.sin(x)*np.cos(y)*np.cos(z)

            w_x = -2*np.exp(-t)*np.sin(y)*np.cos(x)*np.cos(z)
            w_y = -2*np.exp(-t)*np.sin(x)*np.cos(y)*np.cos(z)
            w_z = 2*np.exp(-t)*np.sin(x)*np.sin(y)*np.sin(z)

            p_x = -np.exp(-t)*np.sin(x)*np.cos(y)*np.cos(z)
            p_y = -np.exp(-t)*np.sin(y)*np.cos(x)*np.cos(z)
            p_z = -np.exp(-t)*np.sin(z)*np.cos(x)*np.cos(y)

            return [[u_x, u_y, u_z], [v_x, v_y, v_z], [w_x, w_y, w_z], [p_x, p_y, p_z]]


        def on_train_end(self):
            net = self.model.net   
            device = next(net.parameters()).device
            X = self.X_eval.to(device).double().requires_grad_(True)

            Y = net(X)
            n_out = Y.shape[1]

            grads = []
            for j in range(n_out):
                grad_j = torch.autograd.grad(
                    outputs=Y[:, j],
                    inputs=X,
                    grad_outputs=torch.ones_like(Y[:, j]),
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                grads.append(grad_j.detach().cpu())

            [u_exact, v_exact, w_exact, p_exact] = self.grad_solution(self.np_X_eval)

            u_x_pred = grads[0][:, 0].detach().cpu().numpy()[:, None]
            u_y_pred = grads[0][:, 1].detach().cpu().numpy()[:, None]
            u_z_pred = grads[0][:, 2].detach().cpu().numpy()[:, None]

            v_x_pred = grads[1][:, 0].detach().cpu().numpy()[:, None]
            v_y_pred = grads[1][:, 1].detach().cpu().numpy()[:, None]
            v_z_pred = grads[1][:, 2].detach().cpu().numpy()[:, None]

            w_x_pred = grads[2][:, 0].detach().cpu().numpy()[:, None]
            w_y_pred = grads[2][:, 1].detach().cpu().numpy()[:, None]
            w_z_pred = grads[2][:, 2].detach().cpu().numpy()[:, None]

            p_x_pred = grads[3][:, 0].detach().cpu().numpy()[:, None]
            p_y_pred = grads[3][:, 1].detach().cpu().numpy()[:, None]
            p_z_pred = grads[3][:, 2].detach().cpu().numpy()[:, None]
            
            L2_u_x = np.linalg.norm(u_exact[0] - u_x_pred)
            L2_u_y = np.linalg.norm(u_exact[1] - u_y_pred)
            L2_u_z = np.linalg.norm(u_exact[2] - u_z_pred)

            L2_v_x = np.linalg.norm(v_exact[0] - v_x_pred)
            L2_v_y = np.linalg.norm(v_exact[1] - v_y_pred)
            L2_v_z = np.linalg.norm(v_exact[2] - v_z_pred)

            L2_w_x = np.linalg.norm(w_exact[0] - w_x_pred)
            L2_w_y = np.linalg.norm(w_exact[1] - w_y_pred)
            L2_w_z = np.linalg.norm(w_exact[2] - w_z_pred)

            L2_p_x = np.linalg.norm(p_exact[0] - p_x_pred)
            L2_p_y = np.linalg.norm(p_exact[1] - p_y_pred)
            L2_p_z = np.linalg.norm(p_exact[2] - p_z_pred)

            L2_u_x_exact = np.linalg.norm(u_exact[0])
            L2_u_y_exact = np.linalg.norm(u_exact[1])
            L2_u_z_exact = np.linalg.norm(u_exact[2])

            L2_v_x_exact = np.linalg.norm(v_exact[0])
            L2_v_y_exact = np.linalg.norm(v_exact[1])
            L2_v_z_exact = np.linalg.norm(v_exact[2])

            L2_w_x_exact = np.linalg.norm(w_exact[0])
            L2_w_y_exact = np.linalg.norm(w_exact[1])
            L2_w_z_exact = np.linalg.norm(w_exact[2])

            L2_p_x_exact = np.linalg.norm(p_exact[0])
            L2_p_y_exact = np.linalg.norm(p_exact[1])
            L2_p_z_exact = np.linalg.norm(p_exact[2])

            H1_rel_num_u = L2_u_x ** 2 + L2_u_y ** 2 + L2_u_z ** 2
            H1_rel_den_u = L2_u_x_exact ** 2 + L2_u_y_exact ** 2 + L2_u_z_exact ** 2
            H1_rel_u = np.sqrt(H1_rel_num_u / H1_rel_den_u)

            H1_rel_num_v = L2_v_x ** 2 + L2_v_y ** 2 + L2_v_z ** 2
            H1_rel_den_v = L2_v_x_exact ** 2 + L2_v_y_exact ** 2 + L2_v_z_exact ** 2
            H1_rel_v = np.sqrt(H1_rel_num_v / H1_rel_den_v)

            H1_rel_num_w = L2_w_x ** 2 + L2_w_y ** 2 + L2_w_z ** 2
            H1_rel_den_w = L2_w_x_exact ** 2 + L2_w_y_exact ** 2 + L2_w_z_exact ** 2
            H1_rel_w = np.sqrt(H1_rel_num_w / H1_rel_den_w)

            H1_rel_num_p = L2_p_x ** 2 + L2_p_y ** 2 + L2_p_z ** 2
            H1_rel_den_p = L2_p_x_exact ** 2 + L2_p_y_exact ** 2 + L2_p_z_exact ** 2
            H1_rel_p = np.sqrt(H1_rel_num_p / H1_rel_den_p)

            self.H1_rel_u = H1_rel_u
            self.H1_rel_v = H1_rel_v
            self.H1_rel_w = H1_rel_w
            self.H1_rel_p = H1_rel_p

    #########################################################

    start_time = time.time()

    X_eval = model.data.test_x

    H1_norm = H1_norm_callback(X_eval, train_number)

    for n in range(4):
        model.compile("adam", lr=learning_rate)
        model.train(iterations=7000, display_every=100, callbacks=resampler)

        model.compile("L-BFGS")
        model.train(display_every=100)

    model.compile("adam", lr=learning_rate)
    model.train(iterations=7000, display_every=100, callbacks=resampler)

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100, callbacks=[H1_norm])

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

    l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
    l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_w = dde.metrics.l2_relative_error(w_exact, w_pred)
    l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)

    H1_rel_u = H1_norm.H1_rel_u
    H1_rel_v = H1_norm.H1_rel_v
    H1_rel_w = H1_norm.H1_rel_w
    H1_rel_p = H1_norm.H1_rel_p

    if not os.path.exists(str(train_number)):
        os.makedirs(str(train_number))

    output = [collocation_type, train_time, l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, H1_rel_u, H1_rel_v, H1_rel_w, H1_rel_p, learning_rate, stepsize, number_collocation_points, random_seed, resample_period, number_of_iterations]
    output_file_path = os.path.join(str(train_number), 'output.csv')

    with open(output_file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(output)

if __name__ == "__main__":
    PINN_NS(train_number=0)