
import sys
sys.path.append('asebo')
import numpy as np
from utils.methods import ES_vanilla_gradient, Hess_Aware, run_LP_Hessian_structured
from  utils.nevergrad_functions import F_sphere, F_rastrigin, F_rosenbrock, F_lunacek
import matplotlib.pyplot as plt

#
F = F_rosenbrock

plt.figure(1)

seed = 1
np.random.seed(seed)
initial_pt = np.random.uniform(-2,2,100)

# print("ES vanilla gradient ...")
# res = ES_vanilla_gradient(F, lr=0.001, sigma=0.5, theta_0=initial_pt, num_samples = 100, time_steps = 1000, seed=1)
# plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

print("Hess-Aware ...")
res = Hess_Aware(F, lr = 1, sigma = 0.1, theta_0=initial_pt, num_samples = 10, time_steps = 670, seed=1)
plt.plot(res[3], res[4], label = "Hess_Aware")

print("LP structured Hessian ...")
res = run_LP_Hessian_structured(F, lr=1, sigma=0.05, theta_0=initial_pt, num_samples = 10, time_steps = 1000, seed=seed, alpha=1e-6, beta=0.1, PT_threshold=1)
plt.plot(res[2], res[3], label = "LP_Hessian_structured")


plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("nevergrad function")

plt.show()
