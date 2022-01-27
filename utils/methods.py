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
from utils.utils import Gradient_LP, Gradient_L2, Hessian_LP, Hessian_LP_structured, Hessian_L2_structured, get_PTinverse, orthogonal_gaussian

def gradient_LP_antithetic_estimator(all_rollouts, A, sigma, *args, **kwargs):
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            - sum(all_rollouts[-1])/2
        ) / sigma
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = Gradient_LP(gradient_y, epsilons)
    return g

def invHessian_LP_structured_PTinv_estimator(all_rollouts, A, sigma, PT_threshold=-1):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _, d = A.shape
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (sigma**2)
    var_H_diag, dct_mtx = Hessian_LP_structured(hessian_y, A[:-1, :]/sigma)
    Hinv = dct_mtx @ (np.diag(get_PTinverse(var_H_diag, PT_threshold)) @ dct_mtx)
    return Hinv

def aggregate_rollouts_hessianES(master, A, params):
    """
    For each perturbation (row of A), we do two rollouts, one with the
    original pertubation, one with the negated version.

    Inputs:
        A: a matrix of perturbations to try
    Output:
        all_rollouts: n x 2
        matrix, where n is the number of rows in A
    """
    # F(theta + sigma*epsilons), and F(theta - sigma*epsilons)
    n = A.shape[0]
    all_rollouts = np.zeros([n, 2])
    timesteps = 0
    for i in range(n):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(), 2)
        timesteps += w.timesteps

    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts)  + 1e-8)
    return all_rollouts, timesteps

def HessianES(params, master, gradient_estimator, invhessian_estimator, cov=None):
    """
    Samples from invHessian of previous round
    """
    n_samples = params['num_sensings']
    if cov is None:
        cov = np.identity(master.N)
    mu = np.repeat(0, master.N)

    # A = np.random.multivariate_normal(mu, cov, n_samples)
    np.random.seed(None)
    A = orthogonal_gaussian(master.N, n_samples)
    # A /= np.linalg.norm(A, axis =-1)[:, np.newaxis]
    A *= params["sigma"]
    A = np.vstack([A, mu])  # Adding a reference evaluation

    rollouts, timesteps = aggregate_rollouts_hessianES(master, A, params)

    g = gradient_estimator(rollouts, A, params["sigma"], np.linalg.inv(cov))
    invH = invhessian_estimator(rollouts, A, params["sigma"])
    return (g, invH, n_samples, timesteps)

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
