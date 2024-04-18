import numpy as np
import torch
from collections import deque
import random
import copy

# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

class QFunctionWithNN():
    # qnet
    # targnet
    def __init__(self, 
                 state_dims, 
                 action_dims, 
                 alpha=0.01):
        
        self.qnet = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_dims)
        )

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self, s):
        self.qnet.eval()
        return self.qnet(torch.tensor(s, dtype=torch.float32)).item()
    
    def update(self, observations, y):
        self.qnet.train()    
        self.optimizer.zero_grad()   

        preds = self(torch.tensor(observations, dtype=torch.float32))
        targets = torch.tensor(y, dtype=torch.float32)

        loss = torch.mean((targets - preds) ** 2)
        loss.backward()
        self.optimizer.step()

def dqn(env, 
        epsilon=.1, 
        episodes=10000, 
        gamma=1,
        N=100000, # Replay memory,
        C=50, # Update for target network
        num_samples=64, # minibatch size 
        ):
    
    # Data to save 
    rewards = deque() # Save rewards for each episode
    states = deque() # Save states for each episode
    termination_condition = deque() # Save termination conditions for each episode

    # Initialize replay memory D to capacity N
    D = deque(maxlen=N) # Replay data
    Q = QFunctionWithNN(env.state_dims, env.action_dims)

    # Intialize Q network and target network
    Qtarget = copy.deepcopy(Q)
    update_step = 0 # Counter to update target model

    # Epsilon greedy selection
    def eps_greedy(s):
        # TODO: your code goes here
        if np.random.rand() < epsilon:
            return random.randint(0, len(env.action_space) - 1)
        else:
            return torch.argmax(Q(s)).item()
        
    for episode in range(episodes):
        S = env.reset()
        done = False

        while not done:
            # Select action
            A = eps_greedy(S)

            # Execute action
            Stp1, Rtp1, done, info = env.step(A)

            # Store transition
            D.append((S, A, Rtp1, Stp1, done))

            # Sample from D once enough data is collected
            minibatch = random.sample(D, num_samples)

            y = []
            for i, (S, A, Rtp1, Stp1, done) in enumerate(minibatch):
                if done:
                    y.append(Rtp1)
                else:
                    y.append(Rtp1 + gamma * torch.max(Qtarget(Stp1)).item())
            
            # Update
            Q.update(np.array([ex[0] for ex in minibatch]), np.array(y))

            # Reset Qtarget
            update_step += 1
            if update_step % C == 0:
                Qtarget = copy.deepcopy(Q)

            S = Stp1

    return rewards, states, termination_condition, Q
