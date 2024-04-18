import numpy as np
import matplotlib.pyplot as plt
from environment.world import World
from collections import deque
from dqn import dqn

def main():
    env = World()    

    rewards, states, termination_condition, Q = dqn(env)

if __name__ == "__main__":
    main()
