import numpy as np
import torch
from collections import deque
import random
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

class QFunctionWithNN():
    # qnet
    # targnet
    def __init__(self, state_dims, action_dims, alpha):
        self.qnet = torch.nn.Sequential()
        #self.targnet = torch.nn.Sequential()
        self.optimizer = torch.optim.Adam()

    def update(self, target):
        pass

def dqn(env, 
        epsilon, 
        episodes, 
        gamma,
        N, # Replay memory,
        C, # Update for target network
        num_samples=32, # minibatch size 
        ):
    # Epsilon greedy selection
    def eps_greedy(s):
        # TODO: your code goes here
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q)

    # Initialize replay memory D to capacity N
    D = deque() # Replay data
    Q = QFunctionWithNN()
    Qtarget = QFunctionWithNN()
    Qtarget.load_state_dict(Q.state_dict())
    update_step = 0

    for episode in range(1, episodes):
        S = env.reset()
        done = False

        while not done:
            A = eps_greedy(S)
            Stp1, Rtp1, done, info = env.step(A)
            D.append((S, A, Rtp1, Stp1))

            # Sample from D
            minibatch = random.sample(D, num_samples)
            y = np.zeros(num_samples)

            for i in range(num_samples):
                if done:
                    y[i] = minibatch[2]
                else:
                    y[i] = minibatch[2] + gamma * np.argmax(Qtarget(minibatch[3], ))

            # Update
            Q.update(y)

            # Reset Qtarget
            update_step += 1
            if update_step % C == 0:
                Qtarget.load_state_dict(Q.state_dict())
