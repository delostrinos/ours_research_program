import numpy as np
import matplotlib.pyplot
import random
from scipy.optimize import minimize

# Defining parameters
a_min = 0
a_max = 2
Num_of_a = 30       # number of 'a' values to sample (like 30 radii in the circle problem)
t_max = 1
Num_of_steps = 50   # number of time steps - bigger = smoother Euler
M, N = 3, 3         # polynomial degrees in a and t - START SMALL (recovers 4 terms)
dt = t_max / Num_of_steps    # Euler step size

# Cell 2 - Generate data with Euler's method

# Step 1: sample the 'a' values (uniform, like random_radii in the circle code)
#   -> Num_of_a values between a_min and a_max
# Why do we need a values?
a_values = np.random.uniform(a_min, a_max, Num_of_a)

# Step 2: build the time grid
#   -> time points from 0 to t_max, Num_of_steps of them
time_grid = np.linspace(0, t_max, Num_of_steps)

# Step 3: storage for the flattened triples
# TODO: Why do we need these lists?
a_data = []
t_data = []
x_data = []

# Step 4: loop over each 'a'
#   for each a:
#       start x at 1.0   (initial condition x(0)=1)
#       loop over the time grid:
#           record current (a, t, x) into the three lists
#           Euler update: x = x + dt * a * x
for a in a_values:
    x = 1.0             # initial condition x(0) = 1
    for t in time_grid:
        a_data.append(a)
        t_data.append(t)
        x_data.append(x)
        # Euler update
        x = x + dt * a * x

print(len(a_data), len(t_data), len(x_data))
print("a values:")
print(a_data[0:10])
print("t values:")
print(t_data[0:10])
print("x values:")
print(x_data[0:10])
print(a_data[0], t_data[0], x_data[0])

# ❯ python learn_euler.py 
# 1500 1500 1500
# 1.4042129064426931 0.0 1.0
