import numpy as np
import torch
from collections import deque
import random
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

class QFunctionWithNN():
    # qnet
    # targnet
    def __init__(self, 
                 state_dims, 
                 action_dims, 
                 alpha):
        self.qnet = torch.nn.Sequential()
        self.optimizer = torch.optim.Adam()

    def __call__(self, s):
        self.model.eval()
        return self.model(torch.tensor(s, dtype=torch.float32)).item()
    
    def update(self, observations, y):
        self.qnet.train()    
        self.optimizer.zero_grad()   

        preds = self(torch.tensor(observations, dtype=torch.float32))
        targets = torch.tensor(y, dtype=torch.float32)

        loss = torch.mean((targets - preds) ** 2)
        loss.backward()
        self.optimizer.step()

def dqn(env, 
        epsilon, 
        episodes, 
        gamma,
        N, # Replay memory,
        C, # Update for target network
        num_samples=64, # minibatch size 
        ):
    # Initialize replay memory D to capacity N
    D = deque(maxlen=100000) # Replay data
    Q = QFunctionWithNN()

    # Intialize Q network and target network
    Qtarget = QFunctionWithNN()
    Qtarget.load_state_dict(Q.state_dict())
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

            # Sample from D
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
                Qtarget.load_state_dict(Q.state_dict())

            S = Stp1
