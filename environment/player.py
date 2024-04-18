import pygame
import pygame_gui
import os
import numpy as np

class Player(pygame.sprite.Sprite):
    def __init__(self, image_path, position, angle, x_scale, y_scale, xmin, ymin):
        super().__init__()

        image = pygame.image.load(image_path)

        self.og_image = pygame.transform.scale(image, (50, 33))
        
        self.image = self.og_image
        self.rect = self.image.get_rect()
        self.position = position
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.xmin = xmin
        self.ymin = ymin
        x, y = self.scaled_pos
        self.rect.center = (x, y)
        self.angle = angle

    @property
    def scaled_pos(self):
        scaled_x = self.x_scale * (self.position[0] - self.xmin)
        scaled_y = 800 - self.y_scale * (self.position[1] - self.ymin)

        return scaled_x, scaled_y

    def update(self, position, angle):
        self.image = pygame.transform.rotate(self.og_image, np.rad2deg(angle))
        self.position = position
        scaled_x, scaled_y = self.scaled_pos
        self.rect.center = (scaled_x, scaled_y)