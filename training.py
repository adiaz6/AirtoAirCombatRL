import numpy as np
import matplotlib.pyplot as plt
from environment.world import World
from collections import deque
from dqn2 import dqn

def main():
    env = World()    

    dqn(env)

if __name__ == "__main__":
    main()
