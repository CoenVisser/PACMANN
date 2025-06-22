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


def PINN_Navier_Stokes(train_number, 
                       random_seed_input               = True,
                       amount_collocation_points_input = 700,
                       learning_rate_input             = 1e-3,
                       save_collocation_points_input   = False,
                       collocation_type_input          = "Adam",
                       resample_period_input           = 50,
                       stepsize_input                  = 1e-2,
                       iteration_amount_input          = 5
                       ):
    
    #################### Input Parameters ####################

    if random_seed_input:
        random_seed = random.randint(0, 10000)

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

    # true values
    C1true = 1.0
    C2true = 0.01

    # Load training data
    def load_training_data(num):
        data = np.load("cylinder_nektar_wake.npz")
        U_star = data["U_star"]  # N x 2 x T
        P_star = data["p_star"]  # N x T
        t_star = data["t"]  # T x 1
        X_star = data["X_star"]  # N x 2
        N = X_star.shape[0]
        T = t_star.shape[0]
        # Rearrange Data
        XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
        YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
        TT = np.tile(t_star, (1, N)).T  # N x T
        UU = U_star[:, 0, :]  # N x T
        VV = U_star[:, 1, :]  # N x T
        PP = P_star  # N x T
        x = XX.flatten()[:, None]  # NT x 1
        y = YY.flatten()[:, None]  # NT x 1
        t = TT.flatten()[:, None]  # NT x 1
        u = UU.flatten()[:, None]  # NT x 1
        v = VV.flatten()[:, None]  # NT x 1
        p = PP.flatten()[:, None]  # NT x 1
        # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
        data1 = np.concatenate([x, y, t, u, v, p], 1)
        data2 = data1[:, :][data1[:, 2] <= 7]
        data3 = data2[:, :][data2[:, 0] >= 1]
        data4 = data3[:, :][data3[:, 0] <= 8]
        data5 = data4[:, :][data4[:, 1] >= -2]
        data_domain = data5[:, :][data5[:, 1] <= 2]
        # choose number of training points: num =7000
        idx = np.random.choice(data_domain.shape[0], num, replace=False)
        x_train = data_domain[idx, 0:1]
        y_train = data_domain[idx, 1:2]
        t_train = data_domain[idx, 2:3]
        u_train = data_domain[idx, 3:4]
        v_train = data_domain[idx, 4:5]
        p_train = data_domain[idx, 5:6]
        return [x_train, y_train, t_train, u_train, v_train, p_train]

    # Parameters to be identified
    C1 = dde.Variable(0.0)
    C2 = dde.Variable(0.0)

    def Navier_Stokes_Equation(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        continuity = du_x + dv_y
        x_momentum = du_t + C1 * (u * du_x + v * du_y) + dp_x - C2 * (du_xx + du_yy)
        y_momentum = dv_t + C1 * (u * dv_x + v * dv_y) + dp_y - C2 * (dv_xx + dv_yy)
        return [continuity, x_momentum, y_momentum]
    
    def pde_residual(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        x_momentum = du_t + C1 * (u * du_x + v * du_y) + dp_x - C2 * (du_xx + du_yy)
        y_momentum = dv_t + C1 * (u * dv_x + v * dv_y) + dp_y - C2 * (dv_xx + dv_yy)

        return x_momentum ** 2 + y_momentum ** 2
    
    def residual_gradient_operator(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        continuity = du_x + dv_y
        x_momentum = du_t + C1 * (u * du_x + v * du_y) + dp_x - C2 * (du_xx + du_yy)
        y_momentum = dv_t + C1 * (u * dv_x + v * dv_y) + dp_y - C2 * (dv_xx + dv_yy)

        r = x_momentum ** 2 + y_momentum ** 2
        dr_X = dde.grad.jacobian(r, x, i=0)

        return dr_X

    # Define Spatio-temporal domain
    Lx_min, Lx_max = 1.0, 8.0
    Ly_min, Ly_max = -2.0, 2.0

    space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
    time_domain  = dde.geometry.TimeDomain(0, 7)
    geomtime     = dde.geometry.GeometryXTime(space_domain, time_domain)

    # Get the training data: num = 7000
    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=7000)
    ob_xyt    = np.hstack((ob_x, ob_y, ob_t))
    observe_u = dde.icbc.PointSetBC(ob_xyt, ob_u, component=0)
    observe_v = dde.icbc.PointSetBC(ob_xyt, ob_v, component=1)

    if collocation_type == "Uniform":
        data = dde.data.TimePDE(
            geomtime,
            Navier_Stokes_Equation,
            [observe_u, observe_v],
            num_domain=amount_collocation_points,
            num_boundary=200,
            num_initial=100,
            anchors=ob_xyt, 
            train_distribution="uniform"
        )    
    elif collocation_type == "RAR" or collocation_type == "RAR_D":
        data = dde.data.TimePDE(
            geomtime,
            Navier_Stokes_Equation,
            [observe_u, observe_v],
            num_domain=amount_collocation_points // 2,
            num_boundary=200,
            num_initial=100,
            anchors=ob_xyt,
        )
    else:
        data = dde.data.TimePDE(
            geomtime,
            Navier_Stokes_Equation,
            [observe_u, observe_v],
            num_domain=amount_collocation_points,
            num_boundary=200,
            num_initial=100,
            anchors=ob_xyt,
        )

    layer_size = [3] + [50] * 6 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)

    if save_collocation_points:
    # Save the initial training points
        with open('resampled_points.txt', 'w') as f:
            """
            To obtain the collocation points, use train_x instead of train_x_all since only train_x is ordered, with BC points followed by the collocation points.
            By skipping over these BC points, we find the collocation points. Since BC points might be duplicated, one needs to skip over a multiple of the total number 
            of BC points to obtain the collocation points. https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.pde.PDE.train_x
            """
            f.write(f'0:\n')
            np.savetxt(f, model.data.train_x[21300:, :])

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
                collocation_points_extracted = self.model.data.train_x[21300:, :]
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

            random_sample = geomtime.random_points(self.N_col_points)

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
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[21300:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            X = geomtime.random_points(100000)
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
            
            X = geomtime.random_points(100000)
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
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)

            if self.first_resample:
                collocation_points_extracted = self.model.data.train_x[21300:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            X = geomtime.random_points(100000)
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
                collocation_points = self.model.data.train_x[21300:, :]
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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0])
            inside_time_ids = time_domain.inside(new_collocation_points[:,1])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[21300:, :]
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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0])
            inside_time_ids = time_domain.inside(new_collocation_points[:,1])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[21300:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 3))
            SdX = np.zeros((self.N_col_points, 3))

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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0:2])
            inside_time_ids = time_domain.inside(new_collocation_points[:,2])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points

    class RMSprop(dde.callbacks.Callback):

        def __init__(self, n_iterations, N_col_points, stepsize, beta = 0.999, epsilon = 10e-8, save_points=False, period=50):
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
                collocation_points = self.model.data.train_x[21300:, :]
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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0])
            inside_time_ids = time_domain.inside(new_collocation_points[:,1])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[21300:, :]
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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0])
            inside_time_ids = time_domain.inside(new_collocation_points[:,1])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[21300:, :]
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
            inside_space_ids = space_domain.inside(new_collocation_points[:,0])
            inside_time_ids = time_domain.inside(new_collocation_points[:,1])
            inside_ids = np.logical_and(inside_space_ids, inside_time_ids)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geomtime.random_points(num_random_points)
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
        model.compile("adam", lr=learning_rate, external_trainable_variables=[C1, C2])
        model.train(iterations=7000, display_every=100, callbacks=resampler)

        model.compile("L-BFGS", external_trainable_variables=[C1, C2])
        model.train(display_every=100)

    model.compile("adam", lr=learning_rate, external_trainable_variables=[C1, C2])
    model.train(iterations=7000, display_every=100, callbacks=resampler)

    model.compile("L-BFGS", external_trainable_variables=[C1, C2])
    model.train(display_every=100)

    train_time = time.time() - start_time

    if not os.path.exists(str(train_number)):
        os.makedirs(str(train_number))
    
    C1_final = C1.item()
    C2_final = C2.item()

    print(f'C1: {C1_final}, C2: {C2_final}')

    C1_L2 = np.linalg.norm(C1true - C1_final) / np.linalg.norm(C1true)
    C2_L2 = np.linalg.norm(C2true - C2_final) / np.linalg.norm(C2true)

    output = [collocation_type, train_time, C1_L2, C2_L2, C1_final, C2_final, learning_rate, stepsize, amount_collocation_points, random_seed, resample_period, iteration_amount]
    output_file_path = os.path.join(str(train_number), 'output.csv')

    with open(output_file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(output)

if __name__ == "__main__":
    PINN_Navier_Stokes(train_number=0)