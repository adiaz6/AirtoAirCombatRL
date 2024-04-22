import numpy as np
from environment.world import World
from dqn import dqn
import torch

def main():
    env = World()  
    model = dqn(env)
    torch.save(model.state_dict(), 'model.pt')

if __name__ == "__main__":
    main()
