import sys
import gym
import numpy as np
sys.path.append('./asebo/')
from worker import worker, get_policy

master = get_policy(params)
master.params=np.load("./data/Hopper-v2Linear_h16_lr0.05_num_sensings66_sampleFromInvHFalse_asebo/asebo_params.npy")
test_policy = worker(params, master, np.zeros([1, master.N]), 0)

from gym.wrappers import Monitor
env = Monitor(gym.make(params['env_name']), './video', force=True)
env._max_episode_steps = params['steps']

def play(env, worker):
    state = env.reset()
    while 1:
        action = worker.policy.evaluate(state)
        action = np.clip(action, worker.env.action_space.low[0], worker.env.action_space.high[0])
        action = action.reshape(len(action), )
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
(env, test_policy)