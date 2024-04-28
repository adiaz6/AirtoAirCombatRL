import numpy as np
from environment.world import World
from dqn import dqn
import torch

def main():
    env = World()  
    checkpoint = dqn(env)
    torch.save(checkpoint, './model_baseline.pt')

    #checkpoint = dqn(env, input_model='./model_phase1.pt')
    #torch.save(checkpoint, './model_phase2.pt')

if __name__ == "__main__":
    main()
