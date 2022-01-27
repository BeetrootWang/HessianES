import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import gym
import os
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal
from asebo.worker import get_policy, worker
from asebo.es import ES, aggregate_rollouts
from asebo.optimizers import Adam

def create_data_folder_name(params):
    return params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) \
                    + '_num_sensings' + str(params['num_sensings']) +'_' + 'sigma_'+str(params['sigma'])

def run_HessianES(params, gradient_estimator, invhessian_estimator, master=None, normalize=False):
    params['dir'] = create_data_folder_name(params)
    data_folder = './data/'+params['dir']+'_normalize' + str(normalize) +'_hessianES'
    if not(os.path.exists(data_folder)):
        os.makedirs(data_folder)
    if not master:
        master = get_policy(params)
    print("Policy Dimension: ", master.N)

    test_policy = worker(params, master, np.zeros([1, master.N]), 0)
    reward = test_policy.rollout(train=False)

    n_eps = 0
    n_iter = 1
    ts_cumulative = 0
    ts = [0]
    rollouts = [0]
    rewards = [reward]
    samples = [0]
    cov = np.identity(master.N)
    np.random.seed(params['seed'])
    while ts_cumulative < params['max_ts']:
        params['n_iter'] = n_iter
        g, invH, n_samples, timesteps = HessianES(params, master, gradient_estimator, invhessian_estimator, cov)
        if params['sample_from_invH']:
            cov = -invH
        update_direction = -invH@g
        if normalize:
            update_direction /= (np.linalg.norm(update_direction) / master.N + 1e-8)
        lr = params['learning_rate']
        # Backtracking
        if params['backtracking']:
            update = lr*update_direction
            master.update(update)
            # Evaluate
            test_policy = worker(params, master, np.zeros([1, master.N]), 0)
            reward = test_policy.rollout(train=False)
            count = 0
            while (reward < rewards[-1] + lr*params['alpha']*(g@update_direction)) and lr > 1e-30:
                count += 1
                master.update(-update) # Cancel the previous update first
                lr *= params['beta']
                update = lr*update_direction
                master.update(update)
                # Evaluate
                test_policy = worker(params, master, np.zeros([1, master.N]), 0)
                reward = test_policy.rollout(train=False)
                timesteps += test_policy.timesteps
            # if (reward < rewards[-1] + lr*params['alpha']*(g@update_direction)):
            #     # Do not update
            #     master.update(-update)
            #     reward = rewards[-1]

        else:
            update = lr*update_direction
            master.update(update)
            # Evaluate
            test_policy = worker(params, master, np.zeros([1, master.N]), 0)
            reward = test_policy.rollout(train=False)



        # Book keeping
        ts_cumulative += timesteps
        ts.append(ts_cumulative)

        n_eps += 2 * n_samples
        rollouts.append(n_eps)

        rewards.append(reward)
        samples.append(n_samples)

        print('Iteration: %s, Leanring Rate: %.2e, Time Steps: %.2e, Reward: %.2f, Update Direction Norm: %.2f' %(n_iter, lr, ts_cumulative, reward,  np.linalg.norm(update_direction)))
        n_iter += 1

        out = pd.DataFrame({'Rollouts': rollouts, 'Learning Rate':
        lr, 'Reward': rewards, 'Samples': samples,
        'Timesteps': ts})
        out.to_csv('%s/Seed%s.csv' %(data_folder, params['seed']),
        index=False)

        np.save("{}/hessianES_params_seed{}.npy".format(data_folder, params['seed']),
        master.params)

    return ts, rewards, master
