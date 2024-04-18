import numpy as np
import matplotlib.pyplot as plt
from environment.world import World

if __name__ == "__main__":
    env = World()
    print(env.reset())
    while True:
        env.render()
        env.step(0)
