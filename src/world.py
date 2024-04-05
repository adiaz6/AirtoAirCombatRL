import numpy as np
import matplotlib.pyplot as plt
from agent import Pursuer, Evader

class World(object):
    def __init__(self, x_l, x_u, y_l, y_u, 
                 area_x, area_y, area_r, d, 
                 v_min, v_max, u1_max, u2_max, 
                 start_state_pursuer,
                 start_state_evader,
                 k):
        self.x_l = x_l # x lower limit
        self.x_u = x_u # x upper limit
        self.y_l = y_l # y lower limit
        self.y_u = y_u # y upper limit
        
        self.area_x = area_x # x-coordinates of target area
        self.area_y = area_y # y-coordinates of target area
        self.area_r = area_r # radius of target area

        self.d = d # capture distance

        self.k = 1 # distance reward scaling factor

        self.action_space = list()
        self.pursuer = Pursuer(start_state_pursuer, v_min, v_max, u1_max, u2_max)
        self.evader = Evader(start_state_evader, v_min, v_max, u1_max, u2_max)

    @property
    def pursuer_succeeds(self):
        x_p, y_p = self.pursuer.get_position()
        x_e, y_e = self.evader.get_position()
        return (x_p - x_e) ** 2 + (y_p - y_e) ** 2 <= self.d ** 2

    @property
    def evader_succeeds(self):
        x_p, y_p = self.pursuer.get_position()
        x_e, y_e = self.evader.get_position()
        return  (x_e - self.area_x) ** 2 + (y_e - self.area_y) ** 2 <= self.area_r ** 2
    
    @property
    def evader_cornered(self):
        pass

    @property
    def is_terminal(self):
        return self.pursuer_succeeds or self.evader_succeeds or self.evader_cornered
    
    @property
    def get_state(self):
        return self.pursuer.state
    
    @property
    def get_reward(self):
        if self.evader_succeeds:
            return -10
        elif self.pursuer_succeeds:
            return 10
        elif self.evader_cornered:
            return 5

        # d_tm1: previous distance between both agents
        # d_t: current distance between both agents        
        d_tm1 = 0
        d_t = 0
        return self.k * (d_tm1 - d_t)