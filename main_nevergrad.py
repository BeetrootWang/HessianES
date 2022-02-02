
import sys
sys.path.append('asebo')
import numpy as np
from utils.methods import ES_vanilla_gradient, Hess_Aware
from  utils.nevergrad_functions import F_sphere, F_rastrigin, F_rosenbrock, F_lunacek
import matplotlib.pyplot as plt

#
F = F_sphere

plt.figure(1)

seed = 1
np.random.seed(seed)
initial_pt = np.random.uniform(-2,2,100)

print("ES vanilla gradient ...")
res = ES_vanilla_gradient(F, lr=0.001, sigma=0.5, theta_0=initial_pt, num_samples = 100, time_steps = 1000, seed=1)
plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

print("Hess-Aware ...")
res = Hess_Aware(F, lr = 1, sigma = 0.1, theta_0=initial_pt, num_samples = 100, time_steps = 670, seed=1)
plt.plot(res[3], res[4], label = "Hess_Aware")

# print("LP Hessian ...")
# res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 10, time_steps = 100, seed=1)
# plt.plot(res[3], res[4], label = "LP_Hessian")
#
# print("LP structured Hessian ...")
# res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 10, time_steps = 100, seed=1)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured")
#
# print("LP structured Hessian with PT inverse ...")
# res = LP_Hessian_structured_v2(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 10, time_steps = 100, seed=1)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v2")
#
# print("LP structured Hessian with PT inverse and antithetic samples ...")
# res = LP_Hessian_structured_v3(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 10, time_steps = 100, seed=1)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v3")
#
# print("LP structured Hessian with PT inverse, antithetic samples and backtracking ...")
# res = LP_Hessian_structured_v4(F, alpha = 0.01, sigma = 0.05, theta_0=initial_pt, num_samples = 10, time_steps = 1000, seed=1)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v4")

plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Sphere function")

plt.show()
