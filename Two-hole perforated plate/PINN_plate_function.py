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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

import numpy as np
import torch

import deepxde as dde

def PINN_plate(train_number, 
               random_seed_input               = True,
               number_collocation_points_input = 5000,
               learning_rate_input             = 1e-3,
               save_collocation_points_input   = False,
               collocation_type_input          = "Adam",
               resample_period_input           = 50,
               stepsize_input                  = 1e-6,
               number_of_iterations_input      = 5,
               seed_input                      = 0000
               ):

    #################### Input Parameters ####################

    lmbd = 1.0
    mu = 0.5
    Q = 4.0

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

    hole_1_x = 0.75
    hole_1_y = 0.75
    hole_1_r = 0.15

    hole_2_x = 0.25
    hole_2_y = 0.25
    hole_2_r = 0.15

    #########################################################

    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    dde.config.set_random_seed(random_seed)
    dde.config.set_default_float("float64")

    def pde(x, f):
        E_xx = dde.grad.jacobian(f, x, i=0, j=0)
        E_yy = dde.grad.jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

        S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        S_xy = E_xy * 2 * mu

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

        momentum_x = Sxx_x + Sxy_y 
        momentum_y = Sxy_x + Syy_y 

        stress_x = S_xx - f[:, 2:3]
        stress_y = S_yy - f[:, 3:4]
        stress_xy = S_xy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]
    
    def pde_residual(x, f):
        E_xx = dde.grad.jacobian(f, x, i=0, j=0)
        E_yy = dde.grad.jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

        S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        S_xy = E_xy * 2 * mu

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

        momentum_x = Sxx_x + Sxy_y 
        momentum_y = Sxy_x + Syy_y

        stress_x = S_xx - f[:, 2:3]
        stress_y = S_yy - f[:, 3:4]
        stress_xy = S_xy - f[:, 4:5]

        r_ux = momentum_x ** 2
        r_uy = momentum_y ** 2
        r_sx = stress_x ** 2 
        r_sy = stress_y ** 2 
        r_sxy = stress_xy ** 2
        r_uxy = momentum_x ** 2 + momentum_y ** 2

        return [r_ux, r_uy, r_sx, r_sy, r_sxy, r_uxy]
    
    def residual_gradient_operator(x, f):
        E_xx = dde.grad.jacobian(f, x, i=0, j=0)
        E_yy = dde.grad.jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

        S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        S_xy = E_xy * 2 * mu

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

        momentum_x = Sxx_x + Sxy_y 
        momentum_y = Sxy_x + Syy_y 

        stress_x = S_xx - f[:, 2:3]
        stress_y = S_yy - f[:, 3:4]
        stress_xy = S_xy - f[:, 4:5]

        r = momentum_x ** 2 + momentum_y ** 2 + stress_x ** 2 + stress_y ** 2 + stress_xy ** 2

        dr_X = dde.grad.jacobian(r, x, i=0)

        return dr_X

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1.0)

    def boundary_top(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 1.0)

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)
    
    def on_holes(x, on_boundary):
        if not on_boundary:
            return False
        elif boundary_left(x, True) or boundary_right(x, True) or boundary_top(x, True) or boundary_bottom(x, True):
            return False
        else:
            return True
    
    def traction_free(hole_bcs, y_holes, np_inputs):
        Sxx = y_holes[:, 2:3]
        Syy = y_holes[:, 3:4]
        Sxy = y_holes[:, 4:5]

        np_n = geom.boundary_normal(hole_bcs.detach().cpu().numpy())
        n = torch.as_tensor(np_n, dtype=y_holes.dtype, device=y_holes.device)
        
        nx = n[:, 0:1]
        ny = n[:, 1:2]

        tx = Sxx * nx + Sxy * ny
        ty = Sxy * nx + Syy * ny

        return torch.hstack((tx, ty))
        
    rect = dde.geometry.Rectangle([0, 0], [1, 1])
    hole1 = dde.geometry.Disk([hole_1_x, hole_1_y], hole_1_r)
    hole2 = dde.geometry.Disk([hole_2_x, hole_2_y], hole_2_r)

    two_holes = dde.geometry.CSGUnion(hole1, hole2)
    geom = dde.geometry.CSGDifference(rect, two_holes)

    ux_left_bc = dde.icbc.DirichletBC(
        geom, lambda x: 0, boundary_left, component=0
        )
    uy_bottom_bc = dde.icbc.DirichletBC(
        geom, lambda x: 0, boundary_bottom, component=1
        )
    
    sxx_right_bc = dde.icbc.DirichletBC(
        geom, lambda x: 0, boundary_right, component=2
        )
    sxy_right_bc = dde.icbc.DirichletBC(
        geom, lambda x: 0, boundary_right, component=4
        )

    syy_top_bc = dde.icbc.DirichletBC(
        geom, lambda x: (2 * mu + lmbd) * Q * np.sin(np.pi * x[:, 0:1]), boundary_top, component=3,
        )

    holes_bc = dde.icbc.OperatorBC(
        geom, traction_free, on_holes,
    )

    bcs = [
            ux_left_bc,
            uy_bottom_bc,
            sxx_right_bc,
            sxy_right_bc,
            syy_top_bc,
            holes_bc,
        ]

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=number_collocation_points,
        num_boundary=2000,
        num_test=1000,
        exclusions=np.array([[0.0, 0.0],
                             [1.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 1.0]]),
    )

    net = dde.nn.FNN([2] + [64] * 6 + [5], "sin", "Glorot uniform")
    model = dde.Model(data, net)

    

    if save_collocation_points:
        with open('resampled_points.txt', 'w') as f:
            """
            To obtain the collocation points, use train_x instead of train_x_all since only train_x is ordered, with BC points followed by the collocation points.
            By skipping over these BC points, we find the collocation points. Since BC points might be duplicated, one needs to skip over a multiple of the total number 
            of BC points to obtain the collocation points. https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html#deepxde.data.pde.PDE.train_x
            """
            f.write(f'0:\n')
            np.savetxt(f, model.data.train_x[~geom.on_boundary(model.data.train_x)])

    #################### PACMANN-Adam ####################

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
                collocation_points = self.model.data.train_x[~geom.on_boundary(model.data.train_x)]
                self.first_resample = False
            else:
                collocation_points = self.model.data.train_x_all
            
            VdX = np.zeros((self.N_col_points, 2))
            SdX = np.zeros((self.N_col_points, 2))

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
            inside_ids = geom.inside(new_collocation_points)

            num_random_points = new_collocation_points.shape[0] - np.sum(inside_ids)
            random_sample = geom.random_points(num_random_points)
            new_collocation_points[~inside_ids] = random_sample

            return new_collocation_points 
        
    if collocation_type == "Adam":
        resampler = [Adam(N_col_points=number_collocation_points, save_points=save_collocation_points, period=resample_period, stepsize=stepsize, n_iterations=number_of_iterations)]

    ######################################################
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

    if not os.path.exists(str(train_number)):
        os.makedirs(str(train_number))

    output = [collocation_type, train_time, learning_rate, stepsize, number_collocation_points, random_seed, resample_period, number_of_iterations]
    output_file_path = os.path.join(str(train_number), 'output.csv')

    with open(output_file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(output)

    # Plot the solution and collocation points
    collocation_points = model.data.train_x
    inside_domain = collocation_points[~geom.on_boundary(collocation_points)]
    visualization_x_points = np.linspace(0, 1, 500)
    visualization_y_points = np.linspace(0, 1, 500)
    visualization_X, visualization_Y = np.meshgrid(visualization_x_points, visualization_y_points)
    visualization_points = np.hstack((visualization_X.reshape(-1, 1), visualization_Y.reshape(-1, 1)))
    residual_visualization = model.predict(visualization_points, operator=pde_residual)

    mask_hole1 = (((visualization_X - hole_1_x) ** 2 + (visualization_Y - hole_1_y) ** 2) <= hole_1_r**2)
    mask_hole2 = (((visualization_X - hole_2_x) ** 2 + (visualization_Y - hole_2_y) ** 2) <= hole_2_r**2)

    for idx, residual in enumerate(residual_visualization):
        residual_visualization[idx] = residual.reshape(500, 500)

    for j in range(6):
        residual_visualization[j][mask_hole1] = np.nan
        residual_visualization[j][mask_hole2] = np.nan

    # Squared residual plot
    fig, ax = plt.subplots(figsize=(6,6))

    h = ax.imshow(residual_visualization[5], interpolation='nearest', cmap='rainbow', 
                extent=[0, 1, 0, 1], 
                origin='lower', aspect='auto', norm=LogNorm())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(inside_domain[:,0], inside_domain[:,1], 'k.', markersize = 4, clip_on = False)

    hole1_plot = Circle(
        (hole_1_x, hole_1_y),
        hole_1_r,
        edgecolor="black",
        facecolor="none",
        linewidth=3,
        zorder = 3
    )
    ax.add_patch(hole1_plot)

    hole2_plot = Circle(
        (hole_2_x, hole_2_y),
        hole_2_r,
        edgecolor="black",
        facecolor="none",
        linewidth=3,
        zorder = 3
    )
    ax.add_patch(hole2_plot)

    fs = 15
    ax.set_xlabel('$x$', fontsize = fs)
    ax.set_ylabel('$y$', fontsize = fs)
    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(str(train_number), 'residual_plot.pdf'))


    # Predicted displacement plot
    fig, ax = plt.subplots(figsize=(6,6))
    u_x = model.predict(visualization_points)[:, 0]
    u_y = model.predict(visualization_points)[:, 1]

    solution_visualization = np.sqrt(u_x**2 + u_y**2).reshape(500, 500)

    solution_visualization[mask_hole1] = np.nan
    solution_visualization[mask_hole2] = np.nan

    h = ax.imshow(solution_visualization, interpolation='nearest', cmap='rainbow', 
                extent=[0, 1, 0, 1], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    hole1_plot = Circle(
        (hole_1_x, hole_1_y),
        hole_1_r,
        edgecolor="black",
        facecolor="none",
        linewidth=3,
        zorder = 3
    )
    ax.add_patch(hole1_plot)

    hole2_plot = Circle(
        (hole_2_x, hole_2_y),
        hole_2_r,
        edgecolor="black",
        facecolor="none",
        linewidth=3,
        zorder = 3
    )
    ax.add_patch(hole2_plot)

    fs = 15

    ax.set_xlabel('$x$', fontsize = fs)
    ax.set_ylabel('$y$', fontsize = fs)

    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(str(train_number), 'pred_u_plot.pdf'))

if __name__ == "__main__":
    PINN_plate(train_number=0)     
