import numpy as np
import matplotlib.pyplot as plt
from environment.agent import Pursuer, Evader
from environment.player import Player
import os
import pygame
import time
import random

class World(object):
    def __init__(self,
                 vp_min=0, vp_max=1.2, u1_max=0.3, u2_max=0.8, 
                 ve_min=0, ve_max=1.0, ae_max=0.3, ave_max=1.,
                 x_l=0, x_u=10, y_l=0, y_u=10, d=0.3, dt=0.05, 
                 k1=0.5, k2=0.6, ka=1., kr=0.2, repulsion_radius=1.5,
                 continous_action=False):
        self.x_l = x_l # x lower limit
        self.x_u = x_u # x upper limit
        self.y_l = y_l # y lower limit
        self.y_u = y_u # y upper limit
        
        self.area_r = (self.x_u - self.x_l) / 12 # radius of target area
        self.area_y = (self.y_u - self.y_l) / 2
        self.area_x = 5 * (self.x_u - self.x_l) / 6

        self.d = d # capture distance

        self.k1 = k1   # reward constant
        self.k2 = k2
        self.ka = ka # distance reward scaling factor
        self.kr = kr

        self.dt = dt # time step

        self.vp_min = vp_min
        self.vp_max = vp_max
        self.u1_max = u1_max
        self.u2_max = u2_max
        self.ve_min = ve_min
        self.ve_max = ve_max
        self.ae_max = ae_max
        self.ave_max = ave_max
        
        # Each action choice is [linear_accel_control, angular_accel_control]
        self.action_space = np.array([[0, 0],       # [0 linear accel, 0 angular vel]
                                      [0, u2_max],  # [0 linear accel, max angular vel]
                                      [0, -u2_max], # [0 linear accel, -max angular vel]
                                      [u1_max, 0],  # [max linear accel, 0 angular vel]
                                      [u1_max, u2_max],  # [max linear accel, max angular vel]
                                      [u1_max, -u2_max], # [max linear accel, -max angular vel]
                                      [-u1_max, 0],      # [-max linear accel, 0 angular vel]
                                      [-u1_max, u2_max],   # [-max linear accel, max angular vel]
                                      [-u1_max, -u2_max]]) # [-max linear accel, -max angular vel]
        
        # For pygame rendering
        self.scale_x = 800 / (x_u - x_l)
        self.scale_y = 800 / (y_u - y_l)

        self.initialized = False 

        self.repulsion_radius = repulsion_radius

        # State and action dims
        self.state_dims = 9
        self.action_dims = len(self.action_space)

        # Use continous action or discrete action
        self.continuous_action = continous_action

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
    def pursuer_succeeded(self):
        return self.distance_pe <= self.d

    # Check if evader has reached target area
    @property
    def evader_succeeded(self):
        return self.distance_et <= self.area_r
    
    # Check if evader has been chased beyond target area (outside of boundaries)
    @property
    def evader_cornered(self):
        x, y = self.evader.position
        return x >= self.x_u or x <= self.x_l or y >= self.y_u or y <= self.y_l

    # Check if state is terminal
    @property
    def is_terminal(self):
        return self.pursuer_succeeded or self.evader_succeeded or self.evader_cornered
    
    # Get pursuer state
    @property
    def p_state(self):
        return self.pursuer.state
    
   # Get evader state
    @property
    def e_state(self):
        return self.evader.state
    
    # Get feature vector
    # This is of form [p_x, p_y, p_v, p_theta, e_x, e_y, e_v, evader_distance_to_target_area]
    @property
    def state(self):
        return np.concatenate((self.p_state, self.e_state, np.array([self.distance_et])))
    
    # Get reward
    def get_reward(self, dtm1):
        if self.evader_succeeded:
            return -10, "evader succeeds"
        elif self.pursuer_succeeded:
            return 10, "pursuer succeeds"
        elif self.evader_cornered:
            return 0, "evader cornered"

        # d_tm1: previous distance between both agents
        # d_t: current distance between both agents        
        return self.k1 * (dtm1 - self.distance_pe) + self.k2 / (self.distance_pe - self.distance_et + 1e-3), None
        #return self.k1 * (dtm1 - self.distance_pe) + self.k2 / (self.distance_pe + 1e-3), None
    
    # Take a step according to some action
    # Return new_state, reward, done, info (None)
    def step(self, action):
        if not self.initialized:
            raise ValueError("Environment not initialized. Call reset() before calling step().")
        
        if self.continuous_action:
            action_val = action
        else:
            action_val = self.action_space[action]
                             
        dtm1 = self.distance_pe
        self.pursuer.update_state(action_val)
        self.evader.update_state(np.array([self.pursuer.position[0], self.pursuer.position[1]]))  

        self.pursuer_sprite.update([self.pursuer.position[0], self.pursuer.position[1]], self.pursuer.angle)
        self.evader_sprite.update([self.evader.position[0], self.evader.position[1]], self.evader.angle(self.evader.state))

        reward, info = self.get_reward(dtm1)

        return self.state, reward, self.is_terminal, info
    
    # Initialize everything
    def reset(self):
        self.initialized = True

        pursuer_x = random.uniform(self.x_l, (self.x_u - self.x_l) / 3)
        evader_x = random.uniform(self.x_l, (self.x_u - self.x_l) / 3)

        pursuer_y = random.uniform(self.y_u - (self.y_u - self.y_l) / 3, self.y_u) # Between 5 and 10
        evader_y = random.uniform(self.y_l, self.y_l + (self.y_u - self.y_l) / 3) # Between 0 and 2

        pursuer_angle = random.uniform(-np.pi/2, np.pi/2)
        evader_angle = random.uniform(-np.pi/2, np.pi/2)

        pursuer_v = random.uniform(self.vp_min, 0.1 * self.vp_max) 
        evader_v = random.uniform(self.ve_min, 0.1 * self.ve_max)
        
        pursuer_state = [pursuer_x, pursuer_y, pursuer_v, pursuer_angle]
        evader_state = [evader_x, evader_y, evader_v * np.cos(evader_angle), evader_v * np.sin(evader_angle)]
        #pursuer_state = [self.x_l + (self.x_u - self.x_l)/10, self.y_l + (self.y_u - self.y_l)/10, 0.0, 0.0]
        #evader_state = [self.x_l + (self.x_u - self.x_l)/10, self.y_u - (self.y_u - self.y_l)/10, 0.0, 0.0]

        self.pursuer = Pursuer(pursuer_state, self.x_u, self.x_l, self.y_u, self.y_l, self.vp_min, self.vp_max, self.u1_max, self.u2_max, self.dt)
        self.evader = Evader(np.array([self.area_x, self.area_y]), evader_state, self.ve_min, self.ve_max, self.ae_max, self.ave_max, self.dt, self.ka, self.kr, self.repulsion_radius)
       
        # Rendering 
        self.pursuer_sprite = Player(os.path.join('images', 'pursuersprite.png'), [self.pursuer.position[0], self.pursuer.position[1]], self.pursuer.angle, self.scale_x, self.scale_y, self.x_l, self.y_l)
        self.evader_sprite = Player(os.path.join('images', 'evadersprite.png'), [self.evader.position[0], self.evader.position[1]], self.evader.normalize_angle(self.evader.angle(self.evader.state)), self.scale_x, self.scale_y, self.x_l, self.y_l)
        
        pygame.init()
        pygame.display.set_caption('Target Defense Game')

        self.window_surface = pygame.display.set_mode((800, 800))

        self.bg = pygame.Surface((800, 800))
        self.bg.fill(pygame.Color('#c4e6fb'))

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.pursuer_sprite, self.evader_sprite)

        return self.state

    # Render 
    def render(self):
        time.sleep(0.0)
        self.window_surface.blit(self.bg, (0, 0))
        radius = self.area_r * self.scale_x

        circle = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(circle, (255, 0, 0, 80), (radius, radius), radius)
        self.window_surface.blit(circle, ((self.scale_x * (self.area_x - self.x_l) - radius), (800 - self.scale_y * (self.area_y - self.y_l)) - radius))
        self.all_sprites.draw(self.window_surface)
        pygame.display.update()

    # Normalizing function 
    def normalize(self, state):
        min_val = np.array([self.x_l, self.y_l, self.vp_min, -np.pi/2, self.x_l, self.y_l, self.ve_min, self.ve_min, 0])
        max_val = np.array([self.x_u, self.y_u, self.vp_max, np.pi/2, self.x_u, self.y_u, self.ve_max, self.ve_max, max(self.x_u - self.x_l, self.y_u - self.y_l)])

        return (state - min_val) / (max_val - min_val)
