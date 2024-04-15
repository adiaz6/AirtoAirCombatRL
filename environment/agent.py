import numpy as np

class Pursuer():
    def __init__(self, state, x_u, x_l, y_u, y_l, v_min, v_max, u1_max, u2_max, dt):
        self.v_min = v_min
        self.v_max = v_max
        self.u1_max = u1_max
        self.u2_max = u2_max

        self.dt = dt # Time step

        self.state = state # [x, y, v, theta]

    def update_state(self, a):
        # Integration step
        s_dot = self.dynamics(self.state, a)

        state = np.zeros(4) # [x, y, v, theta]

        # Integrate dynamics here
        # Check for limits

        # Update current state
        self.state = state

    def dynamics(self, s, a):
        # a = [u1 u2], u1: action to control linear acceleration {-u1_max, 0, u1_max}
        #              u2: action to control angular acceleration {-u2_max, 0, u2_max}
        s_dot = np.zeros(4) # [x_dot, y_dot, v_dot, theta_dot]

        s_dot[0] = self.v * np.cos(s[3]) # x_dot
        s_dot[1] = self.v * np.sin(s[3]) # y_dot
        s_dot[2] = a[0] # v_dot
        s_dot[3] = a[1] # theta_dot

        return s_dot
    
    @property
    def position(self):
        return self.state[0], self.state[1]
    
    @property
    def angle(self):
        return self.state[3]
    
class Evader(Pursuer):
    def __init__(self, state, v_min, v_max, u1_max, u2_max, dt, ka=0, kr=0):
        self.v_min = v_min
        self.v_max = v_max
        self.u1_max = u1_max
        self.u2_max = u2_max

        self.dt = dt
        self.state = state # [x, y, vx, vy]

        self.ka = ka 
        self.kr = kr

    def update_state(self):
        # Integration step
        state = np.zeros(4) # [x, y, vx, vy]

        # Integrate dynamics here
        self.state = state

    def dynamics(self, s, area_loc, pursuer_loc):
        # area_loc = [x_coord, y_coord] (location of target area)
        # pursuer_loc = [x_coord, y_coord] (current location of pursuer)
        s_dot = np.zeros(4) # [x_dot, y_dot, vx_dot, vy_dot]

        s_dot[0] = 0
        s_dot[1] = 0
        s_dot[2] = 0 # sum of F_a and F_r (attraction force (vector pointing towards center of area), repulsive force (vector pointing away from pursuer))
        s_dot[3] = 0 
    
    @property
    def position(self):
        return self.state[0], self.state[1]

    @property
    def angle(self):
        return np.degrees(np.arctan2(self.state[3], self.state[2]))
        
        