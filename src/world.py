import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

class World(object):
    def __init__(self, x_l, x_u, y_l, y_u, 
                 area_x, area_y, area_r, d, 
                 v_min, v_max, u1_max, u2_max):
        self.x_l = x_l # x lower limit
        self.x_u = x_u # x upper limit
        self.y_l = y_l # y lower limit
        self.y_u = y_u # y upper limit
        
        self.area_x = area_x # x-coordinates of target area
        self.area_y = area_y # y-coordinates of target area
        self.area_r = area_r # radius of target area

        self.d = d # capture distance

        self.action_space = list()
        self.pursuer = Agent(state, v_min, v_max, u1_max, u2_max)
        self.evader = Agent(state, v_min, v_max, u1_max, u2_max)

    def is_terminal(self, state):
        x_p, y_p = self.pursuer.get_position()
        x_e, y_e = self.evader.get_position()

        # True if pursuer reaches evader
        evader_caught = (x_p - x_e) ** 2 + (y_p - y_e) ** 2 <= self.d ** 2

        # True if evader reaches defense area
        evader_missed = (x_e - self.area_x) ** 2 + (y_e - self.area_y) ** 2 <= self.area_r ** 2

        return evader_caught or evader_missed
    
    def get_state(self, agent):
        return agent.state