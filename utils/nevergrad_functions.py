import numpy as np
import math

def F_sphere(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.sum((theta) ** 2, axis=tuple(range(theta.ndim)[1:]))

def rastrigin(theta):
    return 10*theta.shape[0] + np.sum(theta**2 - 10*np.cos(2*math.pi*theta))

def F_rastrigin(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.apply_along_axis(rastrigin, 1, theta)

def rosenbrock(theta):
    v = 0
    d = theta.shape[0]
    for i in range(d-1):
        v += 100* (theta[i+1] - theta[i])**2 + (theta[i] - 1)**2
    return v

def F_rosenbrock(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.apply_along_axis(rosenbrock, 1, theta)

def lunacek(theta):
    pdim = theta.shape[0]
    s = 1.0 - (1.0 / (2.0 * math.sqrt(pdim + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = - math.sqrt(abs((mu1**2 - 1.0) / s))
    firstSum = secondSum = thirdSum = 0
    for i in range(pdim):
        firstSum += (theta[i]-mu1)**2
        secondSum += (theta[i]-mu2)**2
        thirdSum += 1.0 - math.cos(2*math.pi*(theta[i]-mu1))
    return min(firstSum, 1.0*pdim + s*secondSum)+10*thirdSum

def F_lunacek(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.apply_along_axis(lunacek, 1, theta)