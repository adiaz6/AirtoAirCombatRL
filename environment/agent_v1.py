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
        # Euler integration
        state = self.state + s_dot * self.dt

        # Check for limits
        # Check for velocity limits
        state[2] = np.clip(state[2], self.v_min, self.v_max)
             

        # Check for angle limits
        state[3] = self.normalize_angle(state[3])

        # Update current state
        self.state = state

    def dynamics(self, s, a):
        # a = [u1 u2] # u1: action to control linear acceleration {-u1_max, 0, u1_max}
                      # u2: action to control angular acceleration {-u2_max, 0, u2_max}
        s_dot = np.zeros(4) # [x_dot, y_dot, v_dot, theta_dot]

        s_dot[0] = state[2] * np.cos(s[3]) # x_dot
        s_dot[1] = state[2] * np.sin(s[3]) # y_dot
        s_dot[2] = a[0] # v_dot
        s_dot[3] = a[1] # theta_dot

        #check acceleration limits
        s_dot[2] = np.clip(s_dot[2], -u1_max, u1_max)
        s_dot[3] = np.clip(s_dot[3], -u2_max, u2_max)


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
    def __init__(self, state, v_min, v_max, u1_max, u2_max, dt, ka=0, kr=0):
        self.v_min = v_min
        self.v_max = v_max
        self.u1_max = u1_max
        self.u2_max = u2_max
        self.a_max = a_max

        self.dt = dt
        self.state = state # [x, y, vx, vy]

        self.ka = ka 
        self.kr = kr

    def update_state(self):
        # Integration step
        
        state = np.zeros(4) # [x, y, vx, vy]

        # Integrate dynamics here
        state = self.state + s_dot * self.dt

        #limits on velocity
        self.x = np.cos(s[3])
        self.y = np.sin(s[3])
        state[2] = np.clip(state[2], self.v_min*x, self.v_max*x)
        state[3] = np.clip(state[3], self.v_min*y, self.v_max*y)

        # Calculate angle from velocity vector
        self.new_angle = np.arctan2(state[2], state[3])  # Calculate angle from velocity components (vx, vy)

        self.state = state

    def dynamics(self, s, area_loc, pursuer_loc):
        self.area_loc = [x_coord, y_coord] (location of target area)
        self.pursuer_loc = [x_coord, y_coord] (current location of pursuer)
        s_dot = np.zeros(4) # [x_dot, y_dot, vx_dot, vy_dot]

        s_dot[0] = 0
        s_dot[1] = 0
        s_dot[2] = 0 # sum of F_a and F_r (attraction force (vector pointing towards center of area), repulsive force (vector pointing away from pursuer))
        s_dot[3] = 0 

     def acceleration(self, area_loc, pursuer_loc):
        # Calculate distances
        hx = area_loc[0] - self.state[0]  # Distance of evader from the center of the target area along the x-axis
        hy = area_loc[1] - self.state[1]  # Distance of evader from the center of the target area along the y-axis
        dx = pursuer_loc[0] - self.state[0]  # Distance between the pursuer and the evader along the x-axis
        dy = pursuer_loc[1] - self.state[1]  # Distance between the pursuer and the evader along the y-axis

        # Calculate attraction force
        Fa_x = self.ka * hx
        Fa_y = self.ka * hy

        # Calculate repulsion force
        Fr_x = -self.kr * dx
        Fr_y = -self.kr * dy

        # Calculate acceleration
        a_x = Fa_x + Fr_x
        a_y = Fa_y + Fr_y

        # Limit acceleration
        magnitude = np.sqrt(a_x**2 + a_y**2)
        if magnitude > self.a_max:
            a_x *= self.a_max / magnitude
            a_y *= self.a_max / magnitude

        return np.array([a_x, a_y])
    
    @property
    def position(self):
        return self.state[0], self.state[1]

    @property
    def angle(self):
        return np.degrees(np.arctan2(self.state[3], self.state[2]))