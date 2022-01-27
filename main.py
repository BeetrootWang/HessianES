## test our method

# import packages
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import sys

from utils.methods import run_HessianES, \
                    gradient_antithetic_estimator, gradient_LP_antithetic_estimator, gradient_L2_antithetic_estimator,\
                    invHessian_identity_estimator, invHessian_LP_structured_PTinv_estimator, invHessian_L2_structured_PTinv_estimator

########### Setting up params ##########
params = {
        # 'env_name': 'Swimmer-v2',
        # 'env_name': 'HalfCheetah-v2',
        # 'env_name': 'InvertedPendulum-v2',
        # 'env_name': 'InvertedDoublePendulum-v2',
        # 'env_name': 'Reacher-v2',
        'env_name': 'Hopper-v2',
        'steps':1000,
        'h_dim':32,
        'start':0,
        'max_iter':1000,
        'seed':0,
        'k':140, # ASEBO only?
        'num_sensings':100,
        'log':False,
        'linear':True,
        'threshold':0.995,
        'decay':0.99,
        'learning_rate':0.05,#0.05
        'filename':'',
        'policy':'Toeplitz', # Linear or Toeplitz
        'shift':0,
        'min':10,
        'sigma':1e-1,
        'backtracking':True,
        'alpha': 1e-6,
        'beta': 0.25,
        'sample_from_invH': False,
        'max_ts': 1e7
        }



gradient_estimator = gradient_LP_antithetic_estimator
# gradient_estimator = gradient_antithetic_estimator
# gradient_estimator = gradient_L2_antithetic_estimator

invhessian_estimator = invHessian_LP_structured_PTinv_estimator

# invhessian_estimator =  invHessian_identity_estimator
# params['filename'] = "identity"