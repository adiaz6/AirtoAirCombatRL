import numpy as np
import time

class Agent():
    pass

class Pursuer():
    def __init__(self, state, x_u, x_l, y_u, y_l, v_min, v_max, u1_max, u2_max, dt):
        self.v_min = v_min
        self.v_max = v_max
        self.u1_max = u1_max
        self.u2_max = u2_max
        self.x_u = x_u
        self.x_l = x_l
        self.y_u = y_u
        self.y_l = y_l

        self.dt = dt # Time step

        self.state = state # [x, y, v, theta]

    def update_state(self, a):
        # Integration step
        s_dot = self.dynamics(a)

        state = np.zeros(4) # [x, y, v, theta]

        # Euler integration
        state = self.state + s_dot * self.dt

        # Check for position limits
        state[0] = np.clip(state[0], self.x_l, self.x_u)
        state[1] = np.clip(state[1], self.y_l, self.y_u)

        # Check for velocity limits
        state[2] = np.clip(state[2], self.v_min, self.v_max)

        # Check for angle limits
        state[3] = self.normalize_angle(state[3])

        # Update current state
        self.state = state

    def dynamics(self, a):
        # a = [u1 u2], u1: action to control linear acceleration {-u1_max, 0, u1_max}
        #              u2: action to control angular acceleration {-u2_max, 0, u2_max}
        s_dot = np.zeros(4) # [x_dot, y_dot, v_dot, theta_dot]

        s_dot[0] = self.state[2] * np.cos(self.state[3]) # x_dot
        s_dot[1] = self.state[2] * np.sin(self.state[3]) # y_dot
        s_dot[2] = a[0] # v_dot
        s_dot[3] = a[1] # theta_dot

        return s_dot
    
    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @property
    def position(self):
        return self.state[0], self.state[1]
    
    @property
    def angle(self):
        return self.state[3]
    
class Evader(Pursuer):
    def __init__(self, area_loc, state, v_min, v_max, u1_max, u2_max, dt, ka, kr):
        self.v_min = v_min
        self.v_max = v_max
        self.u1_max = u1_max # mas linear accel
        self.u2_max = u2_max # max angular velocity

        self.dt = dt
        self.state = state # [x, y, vx, vy]

        self.ka = ka # Attraction force constant
        self.kr = kr # Repulsion force constant

        self.area_loc = area_loc

    def update_state(self, pursuer_loc):
        # Integration step
        s_dot = self.dynamics(pursuer_loc)
  
        old_angle = self.angle(self.state)

        # Integrate dynamics here
        state = self.state + s_dot * self.dt

        # Limits on velocity
        v_mag = np.linalg.norm(state[2:])
        new_angle = self.normalize_angle(self.angle(state))

        # Calculate angular velocity
        ang_v = (new_angle - old_angle) / self.dt
        ang_v_throttled = np.clip(ang_v, -self.u2_max, self.u2_max)
        ang_throttled = old_angle + ang_v_throttled * self.dt
        print(np.rad2deg(ang_throttled))
        # Induce velocity limits
        v_throttled = np.clip(v_mag, self.v_min, self.v_max)

        state[2] = v_throttled * np.cos(ang_throttled)
        state[3] = v_throttled * np.sin(ang_throttled)

        self.state = state

    def dynamics(self, pursuer_loc):
        # area_loc = [x_coord, y_coord] (location of target area)
        # pursuer_loc = [x_coord, y_coord] (current location of pursuer)
        a_x, a_y = self.acceleration(pursuer_loc)
        s_dot = np.zeros(4) # [x_dot, y_dot, vx_dot, vy_dot]

        s_dot[0] = self.state[2]
        s_dot[1] = self.state[3]
        s_dot[2] = a_x # sum of F_a and F_r (attraction force (vector pointing towards center of area), repulsive force (vector pointing away from pursuer))
        s_dot[3] = a_y 

        return s_dot
    
    def acceleration(self, pursuer_loc):
        # Calculate distances
        hx = self.area_loc[0] - self.state[0]  # Distance of evader from the center of the target area along the x-axis
        hy = self.area_loc[1] - self.state[1]  # Distance of evader from the center of the target area along the y-axis
        dx = pursuer_loc[0] - self.state[0]  # Distance between the pursuer and the evader along the x-axis
        dy = pursuer_loc[1] - self.state[1]  # Distance between the pursuer and the evader along the y-axis

        # Calculate attraction force
        Fa_x = 0# self.ka / (hx + 0.001)
        Fa_y = 10/(hy+0.001)#self.ka / (hy + 0.001)

        # Calculate repulsion force
        Fr_x = -self.kr / (dx + 0.001)
        Fr_y = -self.kr / (dy + 0.001)

        # Calculate acceleration
        a_x = Fa_x + Fr_x
        a_y = Fa_y + Fr_y

        # Limit acceleration
        magnitude = np.sqrt(a_x**2 + a_y**2)
        if magnitude > self.u1_max:
            a_x *= self.u1_max / magnitude
            a_y *= self.u1_max / magnitude

        return np.array([a_x, a_y])
    
    @property
    def position(self):
        return self.state[0], self.state[1]

    def angle(self, state):
        return np.arctan2(state[3], state[2])    

    @staticmethod
    def normalize_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi 
        
