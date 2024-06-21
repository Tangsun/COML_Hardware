#!/usr/bin/env python
"""
High-level trajectory generator from joystick input

This node generates a trajectory by integrating velocity commands received
from a joystick. This trajectory and its velocity is sent to the ACL outer_loop
module for multirotor trajectory tracking.
"""

import copy
import math

import rospy
import numpy as np

from tf.transformations import euler_from_quaternion
from acl_joy import utils

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose
from snapstack_msgs.msg import Goal, State

class Mode:
    TAKING_OFF = 0
    FLYING = 1
    LANDING = 2
    NOT_FLYING = 3


class Btn:
    A = 0
    B = 1
    X = 2
    Y = 3
    RB = 5
    BACK = 6
    START = 7


class Axes:
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 3
    RIGHT_Y = 4


class ACLJoy:
    def __init__(self):

        # initialize variables for state machine
        self.mode = Mode.NOT_FLYING
        self.flight_initialized = False
        self.takeoff_time = None
        self.initial_alt = 0.0
        self.goal = Goal()
        self.pose = Pose()
        self.pose_frame_id = ''
        self.joy = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'r': 0.0}

        # vehicle name is the namespace
        self.vehname = rospy.get_namespace().replace('/','')

        #
        # Load ROS Parameters
        # 

        # room bounds
        self.bounds = {}
        self.bounds['xmax'] = rospy.get_param('/room_bounds/x_max',  1.0)
        self.bounds['xmin'] = rospy.get_param('/room_bounds/x_min', -1.0)
        self.bounds['ymax'] = rospy.get_param('/room_bounds/y_max',  1.0)
        self.bounds['ymin'] = rospy.get_param('/room_bounds/y_min', -1.0)
        self.bounds['zmax'] = rospy.get_param('/room_bounds/z_max',  1.0)
        self.bounds['zmin'] = rospy.get_param('/room_bounds/z_min',  0.0)

        self.spinup_time = rospy.get_param('cntrl/spinup/time', 0.5)
        self.control_dt = rospy.get_param('~control_dt', 0.01)
        self.vel_based_takeoff = rospy.get_param('~takeoff_smooth_vel', False)
        self.takeoff_vel = rospy.get_param('~takeoff_vel', 0.35)
        self.takeoff_alt = rospy.get_param('~takeoff_alt', 1.0)
        self.takeoff_rel = rospy.get_param('~takeoff_rel', True)
        self.landing_fast_thr = rospy.get_param('~landing_fast_threshold', 0.40)
        self.landing_fast_dec = rospy.get_param('~landing_fast_dec', 0.0035)
        self.landing_slow_dec = rospy.get_param('~landing_slow_dec', 0.001)
        self.max_accel_xy = rospy.get_param('~max_accel_xy', 2.0)
        self.max_accel_z = rospy.get_param('~max_accel_z', 0.8)
        self.expo_xy = rospy.get_param('~expo_xy', 0.5)
        self.expo_z = rospy.get_param('~expo_z', 0.1)
        self.expo_r = rospy.get_param('~expo_r', 0.4)
        self.alpha = rospy.get_param('~alpha', 0.9)
        self.alphar = rospy.get_param('~alphar', 0.9)

        self.joy_kx = rospy.get_param('~kx', 2.0)
        self.joy_ky = rospy.get_param('~ky', 2.0)
        self.joy_kz = rospy.get_param('~kz', 0.5)
        self.joy_kr = rospy.get_param('~kr', 2.0)

        #
        # ROS pub / sub communication
        #

        self.pub_goal = rospy.Publisher('goal', Goal, queue_size=1)

        self.sub_state = rospy.Subscriber("state", State, self.stateCb, queue_size=1)
        self.sub_joystick = rospy.Subscriber("joy", Joy, self.joyCb, queue_size=1)

        self.tim_cntrl = rospy.Timer(rospy.Duration(self.control_dt), self.cntrlCb)

    def stateCb(self, msg):
        self.pose_frame_id = msg.header.frame_id
        self.pose.position = msg.pos
        self.pose.orientation = msg.quat

    def joyCb(self, msg):

        #
        # Capture joystick commands
        #

        if msg.buttons[Btn.A] and self.mode is Mode.NOT_FLYING:
            rospy.loginfo("Waiting for spin up")
            self.mode = Mode.TAKING_OFF

        elif msg.buttons[Btn.X] and (self.mode is Mode.FLYING or self.mode is Mode.TAKING_OFF):
            self.mode = Mode.LANDING

        elif msg.buttons[Btn.B] and self.mode is not Mode.NOT_FLYING:
            rospy.loginfo("Killing")
            self.mode = Mode.NOT_FLYING

        # get velocities from joystick using expo
        vx = utils.expo(self.joy_kx * msg.axes[Axes.RIGHT_Y], self.expo_xy)
        vy = utils.expo(self.joy_ky * msg.axes[Axes.RIGHT_X], self.expo_xy)
        vz = utils.expo(self.joy_kz * msg.axes[Axes.LEFT_Y], self.expo_z)
        r = utils.expo(self.joy_kr * msg.axes[Axes.LEFT_X], self.expo_r)

        self.joy['vx'] = self.alpha * self.joy['vx'] + (1 - self.alpha) * vx
        self.joy['vy'] = self.alpha * self.joy['vy'] + (1 - self.alpha) * vy
        self.joy['vz'] = self.alpha * self.joy['vz'] + (1 - self.alpha) * vz
        self.joy['r'] = self.alphar * self.joy['r'] + (1 - self.alphar) * r

    def cntrlCb(self, event):

        if self.mode is Mode.TAKING_OFF:

            if not self.flight_initialized:
                # capture the initial time
                self.takeoff_time = rospy.Time.now()

                # set the goal to our current position + yaw
                self.goal.p = self.pose.position
                self.goal.v.x = self.goal.v.y = self.goal.v.z = 0
                _, _, self.goal.psi = euler_from_quaternion((
                                            self.pose.orientation.x, self.pose.orientation.y,
                                            self.pose.orientation.z, self.pose.orientation.w))
                self.goal.dpsi = 0

                # The ACL outer loop tracks trajectories and their derivatives.
                # To implement velocity control, we integrate the velocity cmds
                # to obtain the trajectory. This is then position-based control
                # using the goal velocity as the derivative error term of PID.
                self.goal.mode_xy = Goal.MODE_POSITION_CONTROL
                self.goal.mode_z = Goal.MODE_POSITION_CONTROL

                # allow the outer loop to send low-level autopilot commands
                self.goal.power = True

                # what is our initial altitude before takeoff?
                self.initial_alt = self.pose.position.z

                self.flight_initialized = True

                ## Generate a trajectory for velocity-based takeoff
                # what should our desired takeoff altitude be?
                takeoff_alt = self.takeoff_alt + (self.initial_alt if self.takeoff_rel else 0.0)
                self.takeoff_traj = utils.Trajectory(self.initial_alt, takeoff_alt, max_vel=self.takeoff_vel)

            # wait for the motors to spin up before sending a command
            if (rospy.Time.now() - self.takeoff_time >= rospy.Duration(self.spinup_time)):

                if self.vel_based_takeoff:

                    # calculate is the current trajectory time
                    t = rospy.Time.now() - (self.takeoff_time + rospy.Duration(self.spinup_time))

                    # evaluate the takeoff trajectory at the current timestep
                    p, v, a, j = self.takeoff_traj.at_time(t.to_sec())

                    self.goal.p.z = p
                    self.goal.v.z = v
                    self.goal.a.z = a
                    self.goal.j.z = j
                
                THR = 0.100 # threshold for take off check
                takeoff_alt = self.takeoff_traj.pe
                if np.abs(takeoff_alt - self.pose.position.z) < THR: # and self.goal.p.z >= takeoff_alt:
                    self.mode = Mode.FLYING
                    rospy.loginfo("Takeoff complete!")
                else:
                    if not self.vel_based_takeoff:
                        inc = self.takeoff_vel * self.control_dt
                        # Increment the z cmd each timestep for a smooth takeoff.
                        # This is essentially saturating tracking error so actuation is low.
                        self.goal.p.z = utils.clamp(self.goal.p.z + inc, 0.0, takeoff_alt)


        elif self.mode is Mode.FLYING:

            safegoal = self.makeSafeGoal(self.control_dt, **self.joy)
            self.goal.p = safegoal.p
            self.goal.v = safegoal.v
            self.goal.psi = safegoal.psi
            self.goal.dpsi = safegoal.dpsi


        elif self.mode is Mode.LANDING:
            # TODO: allow velocity tweaks from joystick
            self.goal.v.x = self.goal.v.y = self.goal.v.z = 0
            self.goal.dpsi = 0

            # choose between fast and slow landing
            thr = self.landing_fast_thr + (self.initial_alt if self.takeoff_rel else 0.0)
            dec = self.landing_fast_dec if self.pose.position.z > thr else self.landing_slow_dec

            self.goal.p.z = utils.clamp(self.goal.p.z - dec, -0.1, self.bounds['zmax'])

            # TODO: make this closed loop (maybe vel based?)
            if self.goal.p.z == -0.1:
                rospy.loginfo("Landed!")
                self.mode = Mode.NOT_FLYING


        elif self.mode is Mode.NOT_FLYING:
            self.goal.power = False
            self.flight_initialized = False


        self.goal.header.stamp = rospy.Time.now()
        self.goal.header.frame_id = self.pose_frame_id
        self.pub_goal.publish(self.goal)

    def makeSafeGoal(self, dt, vx=0.0, vy=0.0, vz=0.0, r=0.0):
        """Make Safe Goal

        Using the current goal state, this function generates a position+yaw
        trajectory goal by integrating the new desired velocity goal.
        """

        safegoal = Goal()

        #
        # Initial signal conditioning of velocity goals
        #

        # rate limit velocities (to indirectly limit accels)
        vx = utils.rateLimit(dt, -self.max_accel_xy, self.max_accel_xy,
                            self.goal.v.x, vx)
        vy = utils.rateLimit(dt, -self.max_accel_xy, self.max_accel_xy,
                            self.goal.v.y, vy)
        vz = utils.rateLimit(dt, -self.max_accel_z, self.max_accel_z,
                            self.goal.v.z, vz)

        # TODO: limit yawrate?

        #
        # Generate position goals and ensure room bounds are maintained
        #

        # n.b., this logic does not prevent a discontinuous position waypoint
        # from being generated outside the room, e.g., from a mouse click.
        # This is because we are working with vel goals, not position goals.

        # with this goal velocity, what will my predicted next goal pos be?
        # Note: we predict forward using the goal so that the goal position
        # trajectory is smooth. We let the outer loop control worry about
        # actually getting there. The alternative would be to use the current
        # state of the vehicle as the starting position, but this would create
        # a discontinuous goal trajectory.
        nextx = self.goal.p.x + vx * dt
        nexty = self.goal.p.y + vy * dt
        nextz = self.goal.p.z + vz * dt

        # If the predicted position is outside the room bounds, only allow
        # movements that move the vehicle back into the room.
        safegoal.p.x, xclamped = utils.clamp_indicate(nextx,
                                    min(self.bounds['xmin'], self.goal.p.x),
                                    max(self.bounds['xmax'], self.goal.p.x))
        safegoal.p.y, yclamped = utils.clamp_indicate(nexty,
                                    min(self.bounds['ymin'], self.goal.p.y),
                                    max(self.bounds['ymax'], self.goal.p.y))
        safegoal.p.z, zclamped = utils.clamp_indicate(nextz,
                                    min(self.bounds['zmin'], self.goal.p.z),
                                    max(self.bounds['zmax'], self.goal.p.z))

        # If the predicted position is outside the room bounds, zero the vels.
        # But do it so that the accelerations are still bounded.
        if xclamped:
            vx = utils.rateLimit(dt, -self.max_accel_xy, self.max_accel_xy,
                                    self.goal.v.x, 0.0)
        if yclamped:
            vy = utils.rateLimit(dt, -self.max_accel_xy, self.max_accel_xy,
                                    self.goal.v.y, 0.0)
        if zclamped:
            vz = utils.rateLimit(dt, -self.max_accel_z, self.max_accel_z,
                                    self.goal.v.z, 0.0)

        # set linear velocities
        safegoal.v.x = vx
        safegoal.v.y = vy
        safegoal.v.z = vz

        #
        # Generate yaw goal
        #

        # with this goal yawrate, what will my predicted next goal yaw be?
        # Note: same as position---we use the goal yaw instead of current yaw
        # (wrapping doesn't really matter since it becomes a quat in outer loop)
        nextYaw = self.goal.psi + r * dt
        safegoal.psi = utils.wrapToPi(nextYaw)

        # set angular rate
        safegoal.dpsi = r

        return safegoal


if __name__ == '__main__':

    rospy.init_node('joy', anonymous=False)
    ns = rospy.get_namespace()
    try:
        if str(ns) == '/':
            rospy.logfatal("Need to specify namespace as vehicle name.")
            rospy.logfatal("This is tyipcally accomplished in a launch file.")
            rospy.signal_shutdown("no namespace specified")
        else:
            print("Starting joystick teleop node for: {}".format(ns))
            node = ACLJoy()
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
