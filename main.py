import numpy as np
import matplotlib.pyplot as plt
from environment.world import World

if __name__ == "__main__":
    env = World()
    print(env.reset())
    is_terminal = False

    while not is_terminal:
        env.render()
        state, reward, is_terminal, info = env.step(0)
