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

from PINN_Burgers_function import PINN_Burgers

# Define the training parameters
Training_parameters_dictionary = {"random_seed"              : True, 
                                  "amount_collocation_points": 2500,
                                  "learning_rate"            : 1e-3,
                                  "save_collocation_points"  : False,
                                  "collocation_type"         : "Uniform",
                                  "resample_period"          : 50,
                                  "stepsize"                 : 1e-5,
                                  "iteration_amount"         : 15
                                  }

Number_of_runs = 10

"""
Collocation point sampling options: 

Non-adaptive
- "Uniform"
- "Hammersley"
- "Random_resampling"

Adaptive
- "RAR"
- "RAD"
- "RAR_D"

PACMANN
- "Gradient_ascent"
- "Nonlinear_gradient_ascent"
- "RMSprop"
- "Momentum"
- "Adam"
- "Golden_section_search"
"""

# Set the conditions for the training of the PINN
Standard_parameters = list(Training_parameters_dictionary.values())
All_parameters = [Standard_parameters] * Number_of_runs

# Run the PINN
for i in range(len(All_parameters)):

    print("################################################")
    print("Iteration: ", i)
    print("################################################")

    Function_arguments = [i] + All_parameters[i]

    PINN_Burgers(*Function_arguments)