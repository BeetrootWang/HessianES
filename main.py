## test our method

# import packages
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import sys
sys.path.append('asebo')
from asebo.worker import get_policy
from utils.methods import run_HessianES, gradient_LP_antithetic_estimator, invHessian_LP_structured_PTinv_estimator


params = {
        # 'env_name': 'Swimmer-v2',
        'env_name': 'HalfCheetah-v2',
        # 'env_name': 'InvertedPendulum-v2',
        # 'env_name': 'InvertedDoublePendulum-v2',
        # 'env_name': 'Reacher-v2',
        # 'env_name': 'Hopper-v2',
        'steps':1000,
        'h_dim':32,
        'start':0,
        'max_iter':1000,
        'seed':0,
        # 'k':140, # ASEBO only?
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
def auto_param_setups(params):
    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    # params['k'] += -1
    # params['alpha'] = 1 # ASEBO only
    params['zeros'] = False
    master = get_policy(params)
    if params['log']:
        params['num_sensings'] = 4 + int(3 * np.log(master.N))
    if params['linear']:
        params['num_sensings'] = int(2 * master.N)
auto_param_setups(params)

gradient_estimator = gradient_LP_antithetic_estimator
invhessian_estimator = invHessian_LP_structured_PTinv_estimator

for seed in range(10):
    params['seed'] = seed
    master = get_policy(params)

    # params['learning_rate'] = 1
    ts, rewards, master = run_HessianES(params, gradient_estimator, invhessian_estimator, master)
