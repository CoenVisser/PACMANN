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


def PINN_Laplace_Disk(train_number, 
                      random_seed_input               = True,
                      amount_collocation_points_input = 10,
                      learning_rate_input             = 1e-3,
                      save_collocation_points_input   = False,
                      collocation_type_input          = "Adam",
                      resample_period_input           = 50,
                      stepsize_input                  = 1e-2,
                      iteration_amount_input          = 5,
                      seed_input                      = 0000,
                      ):

    #################### Input Parameters ####################

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


    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return dy_xx + dy_yy
    

    def residual_gradient_operator(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)

        r = (dy_xx + dy_yy) ** 2
        dr_X = dde.grad.jacobian(r, x, i=0)

        return dr_X


    def solution(x):
        r = np.sqrt(x[:,0:1] ** 2 + x[:,1:2] ** 2)
        theta = np.arctan2(x[:,1:2], x[:,0:1])
        entire_solution = 0
        order = 4
        for n in range(1,order+1):
            entire_solution += r**n * (np.cos(n * theta) + np.sin(n * theta))
        return entire_solution


    triangle_half_angle = 10.0
    disk = dde.geometry.Disk([0, 0], 1)
    triangle = dde.geometry.Triangle((0,0), (1.1, -np.sin(np.radians(triangle_half_angle))), (1.1, np.sin(np.radians(triangle_half_angle))))
    geom = dde.geometry.csg.CSGDifference(disk, triangle)

    bc = dde.icbc.DirichletBC(
        geom,
        solution,
        lambda x, on_boundary: on_boundary,
    )

    if collocation_type == "Uniform":
        data = dde.data.PDE(
            geometry = geom,
            pde = pde,
            bcs = bc,
            num_domain = amount_collocation_points,
            num_boundary = 100,
            solution = solution,
            train_distribution="uniform",
        )
    elif collocation_type == "RAR" or collocation_type == "RAR_D":
        data = dde.data.PDE(
            geometry = geom,
            pde = pde,
            bcs = bc,
            num_domain = amount_collocation_points // 2,
            num_boundary = 100,
            solution = solution,
            train_distribution="uniform",
        )
    else:
        data = dde.data.PDE(
            geometry = geom,
            pde = pde,
            bcs = bc,
            num_domain = amount_collocation_points,
            num_boundary = 100,
            solution = solution,
        )


    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    if save_collocation_points:
        with open('resampled_points.txt', 'w') as f:
            """
            To obtain the collocation points, use train_x instead of train_x_all since only train_x is ordered, with BC points followed by the collocation points.
            By skipping over these BC points, we find the collocation points. Since BC points might be duplicated, one needs to skip over a multiple of the total number 
            of BC points to obtain the collocation points. https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.pde.PDE.train_x
            """
            f.write(f'0:\n')
            np.savetxt(f, model.data.train_x[200:, :])

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
                collocation_points_extracted = self.model.data.train_x[200:, :]
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

            random_sample = geom.random_points(self.N_col_points)

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
                collocation_points_extracted = self.model.data.train_x[200:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)

            X = geom.random_points(100000)
            f = model.predict(X, operator=pde)
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

            X = geom.random_points(100000)
            Y = np.abs(self.model.predict(X, operator=pde)).astype(np.float64)
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
                collocation_points_extracted = self.model.data.train_x[200:, :]
                data.replace_with_anchors(collocation_points_extracted)
                self.first_resample = False
            
            number_of_resamples = math.floor(35000 / resample_period) # 35000 = iterations of Adam (5 * 7000)

            points_added_per_resample = math.ceil((amount_collocation_points // 2) / number_of_resamples)
            
            X = geom.random_points(100000)
            Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
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
                collocation_points = self.model.data.train_x[200:, :]
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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[200:, :]
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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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

            resample_start = time.time()

            if self.first_resample:
                collocation_points = self.model.data.train_x[200:, :]
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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[200:, :]
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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[200:, :]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 2))
            SdX = np.zeros((self.N_col_points, 2))

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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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
                collocation_points = self.model.data.train_x[200:, :]
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
                    f_l = self.model.predict(x_l, operator=pde)
                    f_r = self.model.predict(x_r, operator=pde)
                else:
                    f_l_prev = f_l
                    f_l = np.where(shifted == 0, f_r, self.model.predict(x_l, operator=pde))
                    f_r = np.where(shifted == 1, f_l_prev, self.model.predict(x_r, operator=pde))

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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
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
        model.compile("adam", lr=learning_rate, metrics=['mean squared error', 'l2 relative error'])
        model.train(iterations=7000, display_every=1, callbacks=resampler)

        model.compile("L-BFGS", metrics=['mean squared error', 'l2 relative error'])
        model.train(display_every=100)

    model.compile("adam", lr=learning_rate, metrics=['mean squared error', 'l2 relative error'])
    model.train(iterations=7000, display_every=100, callbacks=resampler)

    model.compile("L-BFGS", metrics=['mean squared error', 'l2 relative error'])
    losshistory, train_state = model.train(display_every=100)

    train_time = time.time() - start_time

    loss_train = np.array([np.sum(loss) for loss in losshistory.loss_train])
    MSE_error = np.array(losshistory.metrics_test)[:, 0]
    L2_error = np.array(losshistory.metrics_test)[:, 1]

    lowest_loss = np.min(loss_train)
    corresponding_epoch = losshistory.steps[np.argmin(loss_train)]
    corresponding_MSE = MSE_error[np.argmin(loss_train)]
    corresponding_L2 = L2_error[np.argmin(loss_train)]
    print(f'Lowest loss: {lowest_loss} at epoch {corresponding_epoch}, MSE Error = {corresponding_MSE}, L2 Error = {corresponding_L2}')

    if not os.path.exists(str(train_number)):
        os.makedirs(str(train_number))

    output = [collocation_type, train_time, corresponding_MSE, corresponding_L2, np.min(MSE_error), np.min(L2_error), learning_rate, stepsize, amount_collocation_points, random_seed, resample_period, iteration_amount]
    output_file_path = os.path.join(str(train_number), 'output.csv')

    with open(output_file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(output)

if __name__ == "__main__":
    PINN_Laplace_Disk(train_number=0)