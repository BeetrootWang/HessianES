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
from asebo.optimizers import Adam
from utils.utils import Gradient_LP, Gradient_L2, Hessian_LP, Hessian_LP_structured, Hessian_L2_structured, get_PTinverse, orthogonal_gaussian
import copy

# TODO: it is not efficient to store invH or return invH; A better way is to pass the eigenvalues of it

def gradient_LP_antithetic_estimator(all_rollouts, A, sigma, *args, **kwargs):
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            - sum(all_rollouts[-1])/2
        ) / sigma
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = Gradient_LP(gradient_y, epsilons)
    return g

def gradient_antithetic_estimator(all_rollouts, A, sigma, SigmaInv=None):
    _, d = A.shape
    if SigmaInv is None:
        SigmaInv = np.identity(d)
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            )
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = (gradient_y@(epsilons@SigmaInv)) / sigma / len(gradient_y)
    return g

def invHessian_LP_structured_PTinv_estimator(all_rollouts, A, sigma, PT_threshold):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _, d = A.shape
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (sigma**2)
    var_H_diag, dct_mtx = Hessian_LP_structured(hessian_y, A[:-1, :]/sigma)
    Hinv = dct_mtx @ (np.diag(get_PTinverse(var_H_diag, PT_threshold)) @ dct_mtx)
    return Hinv

########################################################################################################################
# functions for Nevergrad

def aggregate_rollouts_hessianES_nevergrad(F, epsilons, sigma, theta_t):
    F_plus = F(theta_t + sigma * epsilons)
    F_minus = F(theta_t - sigma * epsilons)
    all_fnc_values = np.array([F_plus, F_minus]).T
    # import pdb; pdb.set_trace()
    return all_fnc_values

## Benchmarks
def ES_vanilla_gradient(F, lr, sigma, theta_0, num_samples, time_steps, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        # **** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        # **** compute function values ****#
        F_val = F(theta_t + sigma * epsilons)
        count += num_samples
        # **** update theta ****#
        new_theta = theta_t
        F_val = F_val.reshape(1, num_samples)
        update = (F_val @ epsilons).ravel()
        new_theta += lr / (num_samples * sigma) * update
        theta_t = new_theta
        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))
    return theta_t, F(theta_t), None, lst_evals, lst_f

def Hess_Aware(F, lr, sigma, theta_0, num_samples, time_steps, p=1, H_lambda=0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    H = None
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        # **** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d

        # **** compute function values ****#
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_val = np.array([F(theta_t)] * num_samples).ravel()
        count += 2 * num_samples

        if t % p == 0:
            H = np.zeros((d, d))
            eps = np.expand_dims(epsilons, -1)
            eet = eps * np.transpose(eps, (0, 2, 1))
            H_samples = (F_plus.reshape(-1, 1, 1) + F_minus.reshape(-1, 1, 1) - 2 * F_val.reshape(-1, 1, 1)) * eet
            H = H_samples.mean(axis=0) / (2 * sigma ** 2)
            u, s, vh = np.linalg.svd(H)
            H_nh = u @ np.diag(s ** -0.5) @ vh
            H_nh_3d = np.ones((num_samples, d, d)) * H_nh

        # **** update theta: compute g ****#
        Fs = F(theta_t + sigma * np.transpose((H_nh @ np.transpose(epsilons)))) - F(theta_t)
        count += num_samples
        eps = np.expand_dims(epsilons, -1)
        g_samples = H_nh_3d @ eps * Fs.reshape(-1, 1, 1) / sigma
        g = g_samples.mean(axis=0).ravel()

        # **** update theta: the rest ****#
        new_theta = theta_t + lr * g
        theta_t = new_theta

        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

## Our method

def run_LP_Hessian_structured(F, lr, sigma, theta_0, num_samples, time_steps, seed, alpha, beta, PT_threshold):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        eta = lr
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        all_fnc_values = aggregate_rollouts_hessianES_nevergrad(F, epsilons, sigma, theta_t)
        count += 2 * num_samples
        # **** update using Newton's method ****#
        g = gradient_antithetic_estimator(all_fnc_values, epsilons, sigma)
        invH = invHessian_LP_structured_PTinv_estimator(all_fnc_values, epsilons, sigma, PT_threshold)
        update_direction = -invH @ g
        theta_t_ = theta_t + eta * update_direction
        F_t = F(theta_t)
        count += 1

        # backtracking
        cnt = 0
        while F(theta_t_) < (F_t + alpha * eta * np.transpose(g) @ update_direction):
            if cnt >= 30:
                break
            cnt += 1
            eta *= beta
            theta_t_ = theta_t + eta * update_direction


        theta_t = theta_t_
        count += cnt

        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), lst_evals, lst_f

########################################################################################################################
# functions for RL task

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
    np.random.seed(params['seed'])
    A = orthogonal_gaussian(master.N, n_samples)
    # A /= np.linalg.norm(A, axis =-1)[:, np.newaxis]
    A *= params["sigma"]
    A = np.vstack([A, mu])  # Adding a reference evaluation

    rollouts, timesteps = aggregate_rollouts_hessianES(master, A, params)

    g = gradient_estimator(rollouts, A, params["sigma"], np.linalg.inv(cov))
    invH = invhessian_estimator(rollouts, A, params["sigma"], params["PT_threshold"])
    return (g, invH, n_samples, timesteps)

def create_data_folder_name(params):
    return params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) \
                    + '_num_sensings' + str(params['num_sensings']) +'_' + 'sigma_'+str(params['sigma'])\

# Hessian ES
def run_HessianES(params, gradient_estimator, invhessian_estimator, master=None, normalize=False):
    params['dir'] = create_data_folder_name(params)
    data_folder = './data/'+params['dir']+'_normalize' + str(normalize) + '_PT' + str(params['PT_threshold']) \
                  + '_alpha' + str(params['alpha']) + '_beta' + str(params['beta']) \
                  +'_hessianES'
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
        # import pdb; pdb.set_trace()
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
            # while (reward < rewards[-1] + lr*params['alpha_bt']*(g@update_direction)) and lr > 1e-30:
            while (reward < rewards[-1] + lr * params['alpha_bt'] * (g @ update_direction)) and count <= params['max_backtracking']:
                count += 1
                master.update(-update) # Cancel the previous update first
                lr *= params['beta']
                update = lr*update_direction
                master.update(update)
                # Evaluate
                test_policy = worker(params, master, np.zeros([1, master.N]), 0)
                reward = test_policy.rollout(train=False)
                timesteps += test_policy.timesteps
            # if (reward < rewards[-1] + lr*params['alpha_bt']*(g@update_direction)):
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

    return ts, rewards, master, data_folder

def run_HessianES_adap_sigma(params, gradient_estimator, invhessian_estimator, master=None, normalize=False):
    params['dir'] = create_data_folder_name(params)
    data_folder = './data/'+params['dir']+'_normalize' + str(normalize) + '_PT' + str(params['PT_threshold']) \
                  + '_alphabt' + str(params['alpha_bt']) + '_beta' + str(params['beta']) \
                  +'_hessianES_gauss'
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
        # import pdb; pdb.set_trace()
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
            # while (reward < rewards[-1] + lr*params['alpha_bt']*(g@update_direction)) and lr > 1e-30:
            while (reward < rewards[-1] + lr * params['alpha_bt'] * (g @ update_direction)) and count <= params['max_backtracking']:
                count += 1
                master.update(-update) # Cancel the previous update first
                lr *= params['beta']
                update = lr*update_direction
                master.update(update)
                # Evaluate
                test_policy = worker(params, master, np.zeros([1, master.N]), 0)
                reward = test_policy.rollout(train=False)
                timesteps += test_policy.timesteps
            # if (reward < rewards[-1] + lr*params['alpha_bt']*(g@update_direction)):
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

        print('Iteration: %s, Leanring Rate: %.2e, Time Steps: %.2e, Reward: %.2f, Update Direction Norm: %.2f, sigma: %.2f' %(n_iter, lr, ts_cumulative, reward,  np.linalg.norm(update_direction), params['sigma']))
        n_iter += 1
        if n_iter % 5 == 0:
            params['sigma'] = params['sigma']/1

        out = pd.DataFrame({'Rollouts': rollouts, 'Learning Rate':
        lr, 'Reward': rewards, 'Samples': samples,
        'Timesteps': ts})
        out.to_csv('%s/Seed%s.csv' %(data_folder, params['seed']),
        index=False)

        np.save("{}/hessianES_params_seed{}.npy".format(data_folder, params['seed']),
        master.params)

    return ts, rewards, master, data_folder

# Hessian ES + asebo idea
def HessianASEBO(params, gradient_estimator, inv_Hessian_estimator, master, G):
    if params['n_iter'] >= params['k']:
        pca = PCA()
        pca_fit = pca.fit(G)
        var_exp = pca_fit.explained_variance_ratio_
        var_exp = np.cumsum(var_exp)
        n_samples = np.argmax(var_exp > params['threshold']) + 1
        if n_samples < params['min']:
            n_samples = params['min']
        U = pca_fit.components_[:n_samples]
        UUT = np.matmul(U.T, U)
        U_ort = pca_fit.components_[n_samples:]
        UUT_ort = np.matmul(U_ort.T, U_ort)
        alpha = params['alpha']
        if params['n_iter'] == params['k']:
            n_samples = params['num_sensings']
    else:
        UUT = np.zeros([master.N, master.N])
        alpha = 1
        n_samples = params['num_sensings']

    np.random.seed(params['seed'])
    cov = (alpha / master.N) * np.eye(master.N) + ((1 - alpha) / n_samples) * UUT
    # cov = (alpha) * np.eye(master.N) + ((1-alpha) / n_samples * master.N) * UUT
    cov *= params['sigma']
    mu = np.repeat(0, master.N)
    # A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.zeros((n_samples, master.N))
    try:
        l = cholesky(cov, check_finite=False, overwrite_a=True)
        for i in range(n_samples):
            try:
                A[i] = np.zeros(master.N) + l.dot(standard_normal(master.N))
            except LinAlgError:
                A[i] = np.random.randn(master.N)
    except LinAlgError:
        for i in range(n_samples):
            A[i] = np.random.randn(master.N)
    A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]

    # m, timesteps = aggregate_rollouts(master, A, params, n_samples)
    A = np.vstack([A, mu])  # Adding a reference evaluation
    all_rollouts, timesteps = aggregate_rollouts_hessianES(master, A, params)
    g = gradient_estimator(all_rollouts, A, params["sigma"])
    invH = inv_Hessian_estimator(all_rollouts, A, params["sigma"], params['PT_threshold'])
    update_direction = -invH @ g

    # g = np.zeros(master.N)
    # for i in range(n_samples):
    #     eps = A[i, :]
    #     g += eps * m[i]
    # g /= (2 * params['sigma'])

    if params['n_iter'] >= params['k']:
        params['alpha'] = np.linalg.norm(np.dot(g, UUT_ort)) / np.linalg.norm(np.dot(g, UUT))

    return (update_direction, n_samples, timesteps)

def run_hessian_asebo(params, gradient_estimator, inv_Hessian_estimator, master=None):
    params['dir'] = create_data_folder_name(params)
    data_folder = './data/' + params['dir'] + '_hessian_asebo'
    if not (os.path.exists(data_folder)):
        os.makedirs(data_folder)

    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    m = 0
    v = 0

    # params['k'] += -1
    params['alpha'] = 1

    params['zeros'] = False
    if not master:
        master = get_policy(params)
    print("Policy Dimension: ", master.N)

    if params['log']:
        params['num_sensings'] = 4 + int(3 * np.log(master.N))

    if params['k'] > master.N:
        params['k'] = master.N

    test_policy = worker(params, master, np.zeros([1, master.N]), 0)
    reward = test_policy.rollout(train=False)

    n_eps = 0
    n_iter = 1
    ts_cumulative = 0
    ts = [0]
    rollouts = [0]
    rewards = [reward]
    samples = [0]
    alphas = [1]
    G = []

    while ts_cumulative < params['max_ts']:

        params['n_iter'] = n_iter
        gradient, n_samples, timesteps = HessianASEBO(params, gradient_estimator, inv_Hessian_estimator, master, G)
        ts_cumulative += timesteps
        ts.append(ts_cumulative)
        alphas.append(params['alpha'])

        if n_iter == 1:
            G = np.array(gradient)
        else:
            G *= params['decay']
            G = np.vstack([G, gradient])
        n_eps += 2 * n_samples
        rollouts.append(n_eps)

        gradient /= (np.linalg.norm(gradient) / master.N + 1e-8)
        # update = gradient * params['learning_rate']
        update, m, v = Adam(gradient, m, v, params['learning_rate'], n_iter)

        master.update(update)
        test_policy = worker(params, master, np.zeros([1, master.N]), 0)
        reward = test_policy.rollout(train=False)
        rewards.append(reward)
        samples.append(n_samples)

        # print('Iteration: %s, Rollouts: %s, Reward: %s, Alpha: %s, Samples: %s' %(n_iter, n_eps, reward, params['alpha'], n_samples))

        print('Iteration: %s, LR: %.2e, Alpha: %s, Time Steps: %.2e, Reward: %.2f, Update Direction Norm: %.2f' % (
        n_iter, params['learning_rate'], params['alpha'], ts_cumulative, reward, np.linalg.norm(gradient)))
        n_iter += 1

        out = pd.DataFrame({'Rollouts': rollouts, 'Learning Rate':
            params['learning_rate'], 'Reward': rewards, 'Samples': samples,
                            'Timesteps': ts, 'Alpha': alphas})
        out.to_csv('%s/Seed%s.csv' % (data_folder, params['seed']),
                   index=False)

        np.save("{}/asebo_params_seed{}.npy".format(data_folder, params['seed']),
                master.params)
    return ts, rewards, master, data_folder
