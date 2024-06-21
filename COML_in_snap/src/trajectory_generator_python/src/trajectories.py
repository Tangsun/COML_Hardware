from functools import partial
import jax
import jax.numpy as jnp
from utils import spline, random_ragged_spline
from snapstack_msgs.msg import Goal

def create_goal(r_i, dr_i, ddr_i):
    goal = Goal()
    goal.header.frame_id = "world"
    goal.p.x   = r_i[0]
    goal.p.y   = r_i[1]
    goal.p.z   = r_i[2]
    goal.v.x   = dr_i[0]
    goal.v.y   = dr_i[1]
    goal.v.z   = dr_i[2]
    goal.a.x = ddr_i[0]
    goal.a.y = ddr_i[1]
    goal.a.z = ddr_i[2]
    goal.power = True

    goal.psi = 0
    goal.dpsi = 0

    # jerk is set to 0
    goal.j.x  = 0
    goal.j.y  = 0
    goal.j.z  = 0
    return goal

class Spline():
    def __init__(self, num_traj, T, dt, key, xmin_, ymin_, zmin_, xmax_, ymax_, zmax_):
        # Seed random numbers
        self.key = key

        self.xmin_ = xmin_
        self.ymin_ = ymin_
        self.zmin_ = zmin_
        self.xmax_ = xmax_
        self.ymax_ = ymax_
        self.zmax_ = zmax_

        # Generate smooth trajectories
        self.num_traj = num_traj
        self.T = T
        self.dt = dt
        num_knots = 6
        poly_orders = (9, 9, 9)
        deriv_orders = (4, 4, 4)
        min_step = jnp.array([-2, -2, -0.25])
        max_step = jnp.array([2, 2, 0.25])
        min_knot = jnp.array([self.xmin_, self.ymin_, self.zmin_-1]) # z lower bound should be -1.0
        max_knot = jnp.array([self.xmax_, self.ymax_, self.zmax_-1]) # z upper bound should be 1.0

        self.key, *subkeys = jax.random.split(self.key, 1 + self.num_traj)
        subkeys = jnp.vstack(subkeys)
        in_axes = (0, None, None, None, None, None, None, None, None)
        self.t_knots, self.knots, self.coefs = jax.vmap(random_ragged_spline, in_axes)(
            subkeys, self.T, num_knots, poly_orders, deriv_orders,
            min_step, max_step, min_knot, max_knot
        )
        # x_coefs, y_coefs, ϕ_coefs = coefs
        self.r_knots = jnp.dstack(self.knots)

    # Sampled-time simulator
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def simulate(self, ts, t_knots, coefs):
        """TODO: docstring."""
        # Construct spline reference trajectory
        def reference(t):
            x_coefs, y_coefs, z_coefs = coefs
            x = spline(t, t_knots, x_coefs)
            y = spline(t, t_knots, y_coefs)
            z = spline(t, t_knots, z_coefs) + 1.
            x = jnp.clip(x, self.xmin_, self.xmax_)
            y = jnp.clip(y, self.ymin_, self.ymax_)
            z = jnp.clip(z, self.zmin_, self.zmax_)
            r = jnp.array([x, y, z])
            return r

        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
            return r, dr, ddr

        # Simulation loop
        def loop(carry, input_slice):
            t_prev = carry
            t = input_slice

            r, dr, ddr = ref_derivatives(t)
            carry = (t)
            output_slice = (r, dr, ddr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        
        # Run simulation loop
        carry = (t0)
        carry, output = jax.lax.scan(loop, carry, ts[1:])
        r, dr, ddr = output

        # Prepend initial conditions
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))
        ddr = jnp.vstack((ddr0, ddr))

        return r, dr, ddr
    
    def generate_all_trajectories(self):
        self.t = jnp.arange(0, self.T + self.dt, self.dt)  # same times for each trajectory
        # print('t_knots outside: ', t_knots.shape)
        r, dr, ddr = self.simulate(self.t, self.t_knots, self.coefs)

        all_goals = []

        for i in range(self.num_traj):
            goal_i = []
            for r_i, dr_i, ddr_i in zip(r[i], dr[i], ddr[i]):
                goal_i.append(create_goal(r_i, dr_i, ddr_i))
            all_goals.append(goal_i)
                
        return all_goals

class Point():
    def __init__(self, points):
        self.points = points
    
    def generate_all_trajectories(self):
        all_goals = []
        for point in self.points:
            goal_i = [create_goal(point, jnp.zeros(3), jnp.zeros(3))]
            all_goals.append(goal_i)

        return all_goals
    
class Circle():
    def __init__(self, T, dt, radius, center_x, center_y, alt):
        """
        Initializes parameters for generating a circular trajectory.

        Parameters:
        - T (float): Total period of the circle in time units, defining how long it takes to complete one full revolution.
        - dt (float): Time step for trajectory sampling, determining the resolution of the trajectory data.
        - radius (float): Radius of the circle, specifying the distance from the center to any point on the circle.
        - center_x (float): X-coordinate of the circle's center, defining the horizontal position of the circle.
        - center_y (float): Y-coordinate of the circle's center, defining the vertical position of the circle.
        - alt (float): Altitude or constant z-coordinate for the trajectory, maintaining a fixed elevation throughout.
        """
        self.T = T
        self.dt = dt
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.alt = alt

        self.num_traj = 1

        self.r, self.dr, self.ddr = self.populate_circle_trajectory()
    
    def populate_circle_trajectory(self):
        # Angular velocity
        omega = 2 * jnp.pi / self.T

        # Time array
        t = jnp.arange(0, self.T + self.dt, self.dt)
        
        # Angular positions
        theta = omega * t

        # Position calculations
        x = self.center_x + self.radius * jnp.cos(theta)
        y = self.center_y + self.radius * jnp.sin(theta)
        z = self.alt * jnp.ones_like(theta)

        # Velocity calculations
        vx = -self.radius * omega * jnp.sin(theta)
        vy = self.radius * omega * jnp.cos(theta)
        vz = jnp.zeros_like(theta)

        # Acceleration calculations
        ax = -self.radius * omega**2 * jnp.cos(theta)
        ay = -self.radius * omega**2 * jnp.sin(theta)
        az = jnp.zeros_like(theta)

        r = jnp.expand_dims(jnp.vstack((x, y, z)).T, axis=0)
        dr = jnp.expand_dims(jnp.vstack((vx, vy, vz)).T, axis=0)
        ddr = jnp.expand_dims(jnp.vstack((ax, ay, az)).T, axis=0)

        return r, dr, ddr
    
    def generate_all_trajectories(self):
        all_goals = []
        for i in range(self.num_traj):
            goal_i = []
            for r_i, dr_i, ddr_i in zip(self.r[i], self.dr[i], self.ddr[i]):
                goal_i.append(create_goal(r_i, dr_i, ddr_i))
            all_goals.append(goal_i)
        return all_goals

class FigureEight():
    def __init__(self, T, dt, a, b, center_x, center_y, alt):
        """
        Initialize the parameters for the figure-eight trajectory.

        Parameters:
        - T (float): Total time of the trajectory.
        - dt (float): Time step for the trajectory points.
        - a (float): Semi-major axis (x-axis direction amplitude).
        - b (float): Semi-minor axis (y-axis direction amplitude).
        - center_x (float): x-coordinate of the center of the trajectory.
        - center_y (float): y-coordinate of the center of the trajectory.
        - alt (float): Constant altitude of the trajectory.
        """
        self.T = T
        self.dt = dt
        self.a = a
        self.b = b
        self.center_x = center_x
        self.center_y = center_y
        self.alt = alt

        self.num_traj = 1

        self.r, self.dr, self.ddr = self.populate_figure_eight_trajectory()
    
    def populate_figure_eight_trajectory(self):
        # Angular velocity
        omega = 2 * jnp.pi / self.T

        # Time array
        t = jnp.arange(0, self.T + self.dt, self.dt)
        
        # Angular positions
        theta = omega * t

        # Position calculations for figure-eight
        x = self.center_x + self.a * jnp.sin(theta)
        y = self.center_y + self.b * jnp.sin(2 * theta)  # Doubled frequency for figure-eight
        z = self.alt * jnp.ones_like(theta)

        # Velocity calculations
        vx = self.a * omega * jnp.cos(theta)
        vy = self.b * 2 * omega * jnp.cos(2 * theta)
        vz = jnp.zeros_like(theta)

        # Acceleration calculations
        ax = -self.a * omega**2 * jnp.sin(theta)
        ay = -self.b * 4 * omega**2 * jnp.sin(2 * theta)
        az = jnp.zeros_like(theta)

        r = jnp.expand_dims(jnp.vstack((x, y, z)).T, axis=0)
        dr = jnp.expand_dims(jnp.vstack((vx, vy, vz)).T, axis=0)
        ddr = jnp.expand_dims(jnp.vstack((ax, ay, az)).T, axis=0)

        return r, dr, ddr
    
    def generate_all_trajectories(self):
        all_goals = []
        for i in range(self.num_traj):
            goal_i = []
            for r_i, dr_i, ddr_i in zip(self.r[i], self.dr[i], self.ddr[i]):
                goal_i.append(create_goal(r_i, dr_i, ddr_i))
            all_goals.append(goal_i)
        return all_goals