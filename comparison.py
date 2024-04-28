import proportional_navigation as pn
import numpy as np
from environment.world import World
from dqn import dqn, DQN
import torch
import matplotlib.pyplot as plt
import random
from collections import deque
from process_data import comparison_plots

        
def dqn_policy(env, policy_path, num_trials):
    rewards = list()
    success = {"evader succeeds": 0,
               "pursuer succeeds": 0,
               "evader cornered": 0}

    n_actions = env.state_dims
    n_observations = env.action_dims

    policy = torch.load(policy_path)
    policy_net = DQN(n_observations, 9)
    policy_net.load_state_dict(policy['state_dict'])
    policy_net.eval()

    T = 200
    i = 0
    total = 0
    while len(rewards) < num_trials - 1:
        state = env.reset()
        state = env.normalize(state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        print(f'Running trial {i}')

        done = False
        ep_reward = list()
        total_r = 0
        gamma = 0.99
        count = 0
        while not done:
            action = policy_net(state).max(1).indices.view(1, 1)

            state, reward, done, info = env.step(action.item())
            total_r += (gamma ** count) * reward
            ep_reward.append(total_r)

            state = env.normalize(state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            count += 1
        
        if count >= T:
            rewards.append(ep_reward[:T])
            i += 1
        
        success[info] += 1
        total += 1

    success["evader cornered"] = success["evader cornered"]/total
    success["evader succeeds"] = success["evader succeeds"]/total
    success["pursuer succeeds"] = success["pursuer succeeds"]/total

    return rewards, success

if __name__ == "__main__":
    env = World()  
    rewards1, success1 = dqn_policy(env, "model_phase1.pt", 10000)

    env = World()
    rewards2, success2 = dqn_policy(env, "model_phase2.pt", 10000)

    env = World()
    rewards3, success3 = dqn_policy(env, "model_baseline.pt", 10000)

    #np.save('./test_data/rewards_ph1.pkl', rewards1)
    #np.save('./test_data/rewards_ph2.pkl', rewards2)
    #np.save('./test_data/rewards_bl.pkl', rewards3)
    np.save('./test_data/success_ph1.npy', success1)
    np.save('./test_data/success_ph2.npy', success2)
    np.save('./test_data/success_bl.npy', success3)

    comparison_plots(rewards1, rewards2, rewards3, success1, success2, success3)

