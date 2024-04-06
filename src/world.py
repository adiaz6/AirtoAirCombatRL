import numpy as np
import matplotlib.pyplot as plt
from agent import Pursuer, Evader

class World(object):
    def __init__(self, x_l, x_u, y_l, y_u, 
                 area_x, area_y, area_r, d, 
                 v_min, v_max, u1_max, u2_max, 
                 start_state_pursuer,
                 start_state_evader,
                 k, 
                 dt):
        self.x_l = x_l # x lower limit
        self.x_u = x_u # x upper limit
        self.y_l = y_l # y lower limit
        self.y_u = y_u # y upper limit
        
        self.area_x = area_x # x-coordinates of target area
        self.area_y = area_y # y-coordinates of target area
        self.area_r = area_r # radius of target area

        self.d = d # capture distance

        self.k = 1 # distance reward scaling factor

        self.dt = dt # time step

        # Each action choice is [linear_accel_control, angular_accel_control]
        self.action_space = { 0: [0, 0],            # [0 linear accel, 0 angular accel] 
                              1: [0, u2_max],       # [0 linear accel, max angular accel]
                              2: [0, -u2_max],      # [0 linear accel, -max angular accel]
                              3: [u1_max, 0],       # [max linear accel, 0 angular accel]
                              4: [u1_max, u2_max],  # [max linear accel, max angular accel]
                              5: [u1_max, -u2_max], # [max linear accel, -max angular accel]
                              6: [-u1_max, 0],      # [-max linear accel, 0 angular accel]
                              7: [-u1_max, u2_max],   # [-max linear accel, max angular accel]
                              8: [-u1_max, -u2_max] } # [-max linear accel, -max angular accel]
        
        self.possible_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.pursuer = Pursuer(start_state_pursuer, v_min, v_max, u1_max, u2_max)
        self.evader = Evader(start_state_evader, v_min, v_max, u1_max, u2_max)

    # Distance between pursuer and evader
    @property
    def distance_pe(self):
        x_p, y_p = self.pursuer.position
        x_e, y_e = self.evader.position
        return np.sqrt((x_p - x_e) ** 2 + (y_p - y_e) ** 2)
    
    # Distance between evader and target area
    @property
    def distance_et(self):
        x_e, y_e = self.evader.position
        return np.sqrt((x_e - self.area_x) ** 2 + (y_e - self.area_y) ** 2)
    
    # Check if pursuer has reached evader
    @property
    def pursuer_succeeds(self):
        return self.distance_pe <= self.d

    # Check if evader has reached target area
    @property
    def evader_succeeds(self):
        return self.distance_et <= self.area_r
    
    # Check if evader has been chased beyond target area (outside of boundaries)
    @property
    def evader_cornered(self):
        x, y = self.evader.get_position
        return x >= self.x_u or x <= self.x_l or y >= self.y_u or y <= self.y_l

    # Check if state is terminal
    @property
    def is_terminal(self):
        return self.pursuer_succeeds or self.evader_succeeds or self.evader_cornered
    
    # Get pursuer state
    @property
    def state(self):
        return self.pursuer.state
    
    # Get reward
    def get_reward(self, dtm1):
        if self.evader_succeeds:
            return -10
        elif self.pursuer_succeeds:
            return 10
        elif self.evader_cornered:
            return 5

        # d_tm1: previous distance between both agents
        # d_t: current distance between both agents        
        return self.k * (dtm1 - self.distance_pe)
    
    # Take a step according to some action
    # Return new_state, reward, done, info (None)
    def step(self, action):
        dtm1 = self.distance_pe
        self.pursuer.update_state(self.action_space[action])
        self.evader.update_state()  

        return self.pursuer.state, self.get_reward(dtm1), self.is_terminal, None
    
    # Initialize everything randomly
    def reset(self):
        pass

    # Render 
    def render(self):
        pass