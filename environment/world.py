import numpy as np
import matplotlib.pyplot as plt
from agent import Pursuer, Evader
from player import Player
import os
import pygame

class World(object):
    def __init__(self, area_x, area_y, area_r,
                 vp_min=0, vp_max=1.2, u1_max=0.3, u2_max=0.8, 
                 ve_min=0, ve_max=1.2, ae_max=0.3, ave_max=0.8,
                 x_l=0, x_u=10, y_l=0, y_u=10, d=0.3, dt=0.1, k=0.1
                 ):
        self.x_l = x_l # x lower limit
        self.x_u = x_u # x upper limit
        self.y_l = y_l # y lower limit
        self.y_u = y_u # y upper limit
        
        self.area_x = area_x # x-coordinates of target area
        self.area_y = area_y # y-coordinates of target area
        self.area_r = area_r # radius of target area

        self.d = d # capture distance

        self.k = k # distance reward scaling factor

        self.dt = dt # time step
        
        # Each action choice is [linear_accel_control, angular_accel_control]
        self.action_space = np.array([[0, 0],       # [0 linear accel, 0 angular accel]
                                      [0, u2_max],  # [0 linear accel, max angular accel]
                                      [0, -u2_max], # [0 linear accel, -max angular accel]
                                      [u1_max, 0],  # [max linear accel, 0 angular accel]
                                      [u1_max, u2_max],  # [max linear accel, max angular accel]
                                      [u1_max, -u2_max], # [max linear accel, -max angular accel]
                                      [-u1_max, 0],      # [-max linear accel, 0 angular accel]
                                      [-u1_max, u2_max],   # [-max linear accel, max angular accel]
                                      [-u1_max, -u2_max]]) # [-max linear accel, -max angular accel]
        
        self.pursuer = Pursuer(self.start_state_pursuer, vp_min, vp_max, u1_max, u2_max)
        self.evader = Evader(self.start_state_evader, ve_min, ve_max, ae_max, ave_max)

        # For pygame rendering
        self.scale_x = 600 / (x_u - x_l)
        self.scale_y = 600 / (y_u - y_l)

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
        x, y = self.evader.get_position
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
        return np.concatenate(self.p_state, self.e_state, self.distance_et)
    
    # Get reward
    def get_reward(self, dtm1):
        if self.evader_succeeded:
            return -10, "evader succeeds"
        elif self.pursuer_succeeded:
            return 10, "pursuer succeeds"
        elif self.evader_cornered:
            return 5, "evader cornered"

        # d_tm1: previous distance between both agents
        # d_t: current distance between both agents        
        return self.k * (dtm1 - self.distance_pe), None
    
    # Take a step according to some action
    # Return new_state, reward, done, info (None)
    def step(self, action):
        dtm1 = self.distance_pe
        self.pursuer.update_state(self.action_space[action])
        self.evader.update_state()  

        reward, info = self.get_reward(dtm1)

        return self.state, reward, self.is_terminal, info
    
    # Initialize everything randomly
    def reset(self):
        self.start_state_evader = 0
        self.start_state_pursuer = 0
        self.area_r = 0
        self.area_x = 0
        self.area_y = 0

    # Render 
    def render(self):
        pygame.init()
        pygame.display.set_caption('Target Defense Game')

        window_surface = pygame.display.set_mode((600, 600))

        bg = pygame.Surface((600, 600))
        bg.fill(pygame.Color('#c4e6fb'))

        pursuer = Player(os.join('..', 'images', 'pursuersprite.png'))
        evader = Player(os.join('..', 'images', 'evadersprite.png'))

        all_sprites = pygame.sprite.Group()
        all_sprites.add(pursuer, evader)

        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

            window_surface.blit(bg, (0, 0))
            radius = self.area_r * self.x_scale

            circle = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(circle, (255, 0, 0, 80), (radius, radius), radius)
            window_surface.blit(circle, (self.x_scale * (self.area_x - self.x_l), 600 - self.y_scale * (self.area_y - self.y_l)))
            all_sprites.draw(window_surface)
            pygame.display.update()