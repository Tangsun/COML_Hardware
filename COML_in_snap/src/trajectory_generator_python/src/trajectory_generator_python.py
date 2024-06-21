#!/usr/bin/env python3

import numpy as np
import datetime
import rospy
import rospkg
import math
import os
import pickle
import jax
from geometry_msgs.msg import Pose, Twist, Vector3
from snapstack_msgs.msg import State, Goal, QuadFlightMode, Wind, AttitudeCommand
from structs import FlightMode
from threading import Event

from helpers import quat2yaw, saturate, simpleInterpolation, wrap, start_rosbag_recording, stop_rosbag_recording
from trajectories import Circle, FigureEight, Point, Spline
from wind import WindSim

class TrajectoryGenerator:
    def __init__(self):
        self.traj_type = 'circle' # 'spline', 'point', 'circle', 'figure_eight'
        self.auto = False # automatically start and stop following trajectories?
        self.wind_type = 'None' # None, 'sine', 'random_constant', 'random_sine', int/float
        self.num_traj = 1
        self.rosbag = True

        self.T = 30
        self.seed = 2
        self.key = jax.random.PRNGKey(self.seed)

        self.freq = rospy.get_param('~pub_freq', default=None)
        self.dt_ = 1.0 / self.freq

        self.vel_initpos_ = rospy.get_param("~vel_initpos")
        self.vel_take_ = rospy.get_param("~vel_take")
        self.vel_land_fast_ = rospy.get_param("~vel_land_fast")
        self.vel_land_slow_ = rospy.get_param("~vel_land_slow")
        self.vel_yaw_ = rospy.get_param("~vel_yaw")

        self.dist_thresh_ = rospy.get_param("~dist_thresh")
        self.yaw_thresh_ = rospy.get_param("~yaw_thresh")

        self.margin_takeoff_outside_bounds_ = rospy.get_param("~margin_takeoff_outside_bounds")

        # Room bounds
        self.xmin_ = rospy.get_param("~/room_bounds/x_min")
        self.xmax_ = rospy.get_param("~/room_bounds/x_max")
        self.ymin_ = rospy.get_param("~/room_bounds/y_min")
        self.ymax_ = rospy.get_param("~/room_bounds/y_max")
        self.zmin_ = rospy.get_param("~/room_bounds/z_min")
        self.zmax_ = rospy.get_param("~/room_bounds/z_max")


        self.flight_mode_ = FlightMode.GROUND
        self.pose_ = Pose()
        self.vel_ = Twist()
        self.goal_ = Goal()
        self.init_pos_ = Goal()
        self.wind_ = Wind()
        self.att_cmd_ = AttitudeCommand()

        self.record = False

        self.rosbag_proc = None

        self.state_initialized = Event()

        # Subscribe and publish
        rospy.Subscriber('/globalflightmode', QuadFlightMode, self.mode_cb)
        rospy.Subscriber('state', State, self.state_cb)
        rospy.Subscriber('attcmd', AttitudeCommand, self.cmd_cb)
 
        self.state_initialized.wait()  # To ensure that the state has been received

        self.pub_timer_ = rospy.Timer(rospy.Duration(self.dt_), self.pub_cb)
        self.pub_goal_ = rospy.Publisher('goal',Goal, queue_size=1)
        # self.pub_wind_ = rospy.Publisher('wind',Wind, queue_size=1)

        self.traj_goals_full_ = self.generate_trajectory()
        self.winds_full_ = self.generate_wind()
        self.index = 0

        self.reset_goal()
        self.goal_.p.x = self.pose_.position.x
        self.goal_.p.y = self.pose_.position.y
        self.goal_.p.z = self.pose_.position.z

        self.reset_wind()

        self.q = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))
        self.dq = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))
        self.u = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))
        self.r = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))
        self.dr = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))
        self.quat = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 4))
        self.omega = np.zeros((self.num_traj, int(self.T/self.dt_)+1, 3))

        rospy.loginfo("Successfully launched trajectory generator node.")

        if self.auto:
            self.take_off()

    def take_off(self):
        # Check inside safety bounds, and don't let the takeoff happen if outside them
        xmin = self.xmin_ - self.margin_takeoff_outside_bounds_
        ymin = self.ymin_ - self.margin_takeoff_outside_bounds_
        xmax = self.xmax_ + self.margin_takeoff_outside_bounds_
        ymax = self.ymax_ + self.margin_takeoff_outside_bounds_
        if self.pose_.position.x < xmin or self.pose_.position.x > xmax or \
        self.pose_.position.y < ymin or self.pose_.position.y > ymax:
            rospy.logwarn("Can't take off: the vehicle is outside the safety bounds.")
            return

        # Takeoff initializations
        self.init_pos_.p.x = self.pose_.position.x
        self.init_pos_.p.y = self.pose_.position.y
        self.init_pos_.p.z = self.pose_.position.z
        # set the goal to our current position + yaw
        self.reset_goal()
        self.goal_.p.x = self.init_pos_.p.x
        self.goal_.p.y = self.init_pos_.p.y
        self.goal_.p.z = self.init_pos_.p.z
        self.goal_.psi = quat2yaw(self.pose_.orientation)

        self.reset_wind()

        # Take off
        self.flight_mode_ = FlightMode.TAKING_OFF
        rospy.loginfo("Taking off...")
        # then it will switch automatically to HOVERING

        # switch on motors after flight_mode changes, to avoid timer callback setting power to false
        self.goal_.power = True
    
    def mode_cb(self, msg):
    # FSM transitions:
    # Any state        --ESTOP--> [kill motors] and switch to ground mode
    # On the ground    --START--> Take off and then hover
    # Hovering         --START--> Go to init pos of the trajectory
    # Init pos of traj --START--> follow the generated trajectory
    # Init pos of traj --END--> switch to hovering where the drone currently is
    # Traj following   --END--> change the traj goals vector to a braking trajectory and then switch to hovering
    # Hovering         --END--> Go to the initial position, land, and then switch to ground mode

    # Behavior selector button to mode mapping
    # START -> GO   (4)
    # END   -> LAND (2)
    # ESTOP -> KILL (6)
        if msg.mode == msg.KILL:
            self.goal_.power = False
            self.goal_.header.stamp = rospy.Time.now()
            self.pub_goal_.publish(self.goal_)
            self.flight_mode_ = FlightMode.GROUND
            self.reset_goal()
            if self.rosbag_proc:
                stop_rosbag_recording(self.rosbag_proc)
            rospy.loginfo("Motors killed, switched to GROUND mode.")
            return
        elif self.flight_mode_ == FlightMode.GROUND and msg.mode == msg.GO:
            self.take_off()
        elif self.flight_mode_ == FlightMode.HOVERING and msg.mode == msg.GO:
            self.traj_goals_ = self.traj_goals_full_[self.index]
            # self.wind_ = self.winds_full_[self.index]
            self.flight_mode_ = FlightMode.INIT_POS_TRAJ
            print(f"Going to the initial position of trajectory {self.index+1}...")
        elif self.flight_mode_ == FlightMode.INIT_POS_TRAJ and msg.mode == msg.GO:
            if self.rosbag and not self.rosbag_proc:
                self.rosbag_proc = start_rosbag_recording('-a')
            # Start following the generated trajectory if close to the init pos (in 2D)
            dist_to_init = math.sqrt(pow(self.traj_goals_[0].p.x - self.pose_.position.x, 2) +
                                    pow(self.traj_goals_[0].p.y - self.pose_.position.y, 2) +
                                    pow(self.traj_goals_[0].p.z - self.pose_.position.z, 2))
            delta_yaw = self.traj_goals_[0].psi - quat2yaw(self.pose_.orientation)
            delta_yaw = wrap(delta_yaw)
            if dist_to_init > self.dist_thresh_ or abs(delta_yaw) > self.yaw_thresh_:
                rospy.loginfo("Can't switch to the generated trajectory following mode, too far from the init pos")
                return
            self.pub_index_ = 0
            self.flight_mode_ = FlightMode.TRAJ_FOLLOWING
            # Start wind
            self.traj_winds_ = self.winds_full_[self.index]
            self.record = True
            rospy.loginfo(f"Following trajectory {self.index+1}...")
        elif self.flight_mode_ == FlightMode.INIT_POS_TRAJ and msg.mode == msg.LAND:
            # Change mode to hover wherever the robot was when we clicked "END"
            # Need to send a current goal with 0 vel bc we could be moving to the init pos of traj
            self.reset_goal()
            self.goal_.p.x = self.pose_.position.x
            self.goal_.p.y = self.pose_.position.y
            self.goal_.p.z = self.alt_
            self.goal_.psi = quat2yaw(self.pose_.orientation)
            self.goal_.header.stamp = rospy.Time.now()
            self.pub_goal_.publish(self.goal_)
            self.flight_mode_ = FlightMode.HOVERING
            rospy.loginfo("Switched to HOVERING mode")
        elif self.flight_mode_ == FlightMode.TRAJ_FOLLOWING and msg.mode == msg.LAND:
            # Generate a braking trajectory. Then, we will automatically switch to hover when done
            # self.generate_stop_traj(self.traj_goals_, self.index_msgs_, self.pub_index_)
            self.flight_mode_ = FlightMode.HOVERING
            rospy.loginfo("Switched to HOVERING mode")
        elif self.flight_mode_ == FlightMode.TAKING_OFF and msg.mode == msg.LAND:
            # Generate a braking trajectory. Then, we will automatically switch to hover when done
            # self.generate_stop_traj(self.traj_goals_, self.index_msgs_, self.pub_index_)
            self.flight_mode_ = FlightMode.HOVERING
            rospy.loginfo("Switched to HOVERING mode")
        elif self.flight_mode_ == FlightMode.HOVERING and msg.mode == msg.LAND:
            # go to the initial position
            self.flight_mode_ = FlightMode.INIT_POS
            rospy.loginfo("Switched to INIT_POS mode")
    
    def state_cb(self, msg):
        self.pose_.position.x = msg.pos.x
        self.pose_.position.y = msg.pos.y
        self.pose_.position.z = msg.pos.z
        self.pose_.orientation = msg.quat

        self.vel_.linear.x = msg.vel.x
        self.vel_.linear.y = msg.vel.y
        self.vel_.linear.z = msg.vel.z
        self.vel_.angular.x = msg.w.x
        self.vel_.angular.y = msg.w.y
        self.vel_.angular.z = msg.w.z

        if not self.state_initialized.is_set():
            self.alt_ = msg.pos.z + 1.0 # set alt_ to 1m above initial z position
            self.state_initialized.set()
    
    def cmd_cb(self, msg):
        self.att_cmd_.F_W.x = msg.F_W.x
        self.att_cmd_.F_W.y = msg.F_W.y
        self.att_cmd_.F_W.z = msg.F_W.z

        self.att_cmd_.q.w = msg.q.w
        self.att_cmd_.q.x = msg.q.x
        self.att_cmd_.q.y = msg.q.y
        self.att_cmd_.q.z = msg.q.z

        self.att_cmd_.w.x = msg.w.x
        self.att_cmd_.w.y = msg.w.y
        self.att_cmd_.w.z = msg.w.z
    
    def pub_cb(self, event):
        # on ground
        if self.flight_mode_ == FlightMode.GROUND:
            self.goal_.power = False  # Not needed but just in case, for safety

        # taking off
        elif self.flight_mode_ == FlightMode.TAKING_OFF:
            # TODO: spinup time

            takeoff_alt = self.alt_  # Don't add init alt bc the traj is generated with z = alt_
            # If close to the takeoff_alt, switch to HOVERING
            if abs(takeoff_alt - self.pose_.position.z) < 0.2 and self.goal_.p.z >= takeoff_alt:
                if self.auto:
                    self.traj_goals_ = self.traj_goals_full_[self.index]
                    self.flight_mode_ = FlightMode.INIT_POS_TRAJ
                else:
                    self.flight_mode_ = FlightMode.HOVERING
                print(f"Take off completed")
            else:
                # Increment the z cmd each timestep for a smooth takeoff.
                # This is essentially saturating tracking error so actuation is low.
                self.goal_.p.z = saturate(self.goal_.p.z + self.vel_take_ * self.dt_, 0.0, takeoff_alt)

        # go to initial position of trajectory
        elif self.flight_mode_ == FlightMode.INIT_POS_TRAJ:
            finished = False
            self.goal_, finished = simpleInterpolation(self.goal_, self.traj_goals_[0], self.traj_goals_[0].psi, 
                self.vel_initpos_, self.vel_yaw_, self.dist_thresh_, 
                self.yaw_thresh_, self.dt_, finished)
            if (self.auto) and finished: 
                if self.rosbag and not self.rosbag_proc:
                    self.rosbag_proc = start_rosbag_recording('-a')
                    # Start following the generated trajectory if close to the init pos (in 2D)
                dist_to_init = math.sqrt(pow(self.traj_goals_[0].p.x - self.pose_.position.x, 2) +
                                        pow(self.traj_goals_[0].p.y - self.pose_.position.y, 2) +
                                        pow(self.traj_goals_[0].p.z - self.pose_.position.z, 2))
                delta_yaw = self.traj_goals_[0].psi - quat2yaw(self.pose_.orientation)
                delta_yaw = wrap(delta_yaw)
                if dist_to_init > self.dist_thresh_ or abs(delta_yaw) > self.yaw_thresh_:
                    rospy.loginfo("Can't switch to the generated trajectory following mode, too far from the init pos")
                    return
                self.pub_index_ = 0
                self.flight_mode_ = FlightMode.TRAJ_FOLLOWING
                # Start wind
                self.traj_winds_ = self.winds_full_[self.index]
                self.record = True
                rospy.loginfo(f"Following trajectory {self.index+1}...")

        # follow trajectory
        elif self.flight_mode_ == FlightMode.TRAJ_FOLLOWING:
            self.goal_ = self.traj_goals_[self.pub_index_]
            self.wind_ = self.traj_winds_[self.pub_index_]
            # TODO: RECORD DATA FOR STATE AND GOAL
            self.record_data()
            self.pub_index_ += 1
            if self.pub_index_ == len(self.traj_goals_):
                self.index += 1
                if self.index == len(self.traj_goals_full_):
                    if self.auto:
                        self.flight_mode_ = FlightMode.INIT_POS
                        self.reset_wind()
                        rospy.loginfo("All trajectories completed, going to initial position")
                        return
                    else:
                        self.reset_goal()
                        self.goal_.p.x = self.pose_.position.x
                        self.goal_.p.y = self.pose_.position.y
                        self.goal_.p.z = self.alt_
                        self.goal_.psi = quat2yaw(self.pose_.orientation)
                        self.goal_.header.stamp = rospy.Time.now()
                        self.pub_goal_.publish(self.goal_)
                        self.flight_mode_ = FlightMode.HOVERING
                        rospy.loginfo("All trajectories completed, switched to HOVERING mode")
                        return
                self.traj_goals_ = self.traj_goals_full_[self.index]
                self.reset_wind()
                self.record = False
                # self.index_msgs_ = self.index_msgs_full_
                self.flight_mode_ = FlightMode.INIT_POS_TRAJ
                print(f"Trajectory {self.index} completed, going to the initial position of trajectory {self.index+1}...")

        # go to initial pos?
        elif self.flight_mode_ == FlightMode.INIT_POS:
            # Go to init_pos_ but with altitude alt_ and current yaw
            finished = False
            dest = self.init_pos_
            dest.p.z = self.alt_
            self.goal_, finished = simpleInterpolation(self.goal_, dest, self.goal_.psi, self.vel_initpos_,
                                            self.vel_yaw_, self.dist_thresh_, self.yaw_thresh_,
                                            self.dt_, finished)
            if finished:  # land when close to the init pos
                # self.reset_wind()
                self.flight_mode_ = FlightMode.LANDING
                if self.rosbag_proc:
                    stop_rosbag_recording(self.rosbag_proc)
                print("Landing...")

        # If landing, decrease alt until we reach ground (and switch to ground)
        # The goal was already set to our current position + yaw when hovering
        elif self.flight_mode_ == FlightMode.LANDING:
            # Choose between fast and slow landing
            vel_land = self.vel_land_fast_ if self.pose_.position.z > (self.init_pos_.p.z + 0.4) else self.vel_land_slow_
            self.goal_.p.z -= vel_land * self.dt_

            if self.goal_.p.z < 0:  # Don't use init alt here. It's safer to try to land to the ground
                # Landed, kill motors
                self.goal_.power = False
                self.flight_mode_ = FlightMode.GROUND
                print("Landed")

        # Apply safety bounds
        self.goal_.p.x = saturate(self.goal_.p.x, self.xmin_, self.xmax_)  # val, low, high
        self.goal_.p.y = saturate(self.goal_.p.y, self.ymin_, self.ymax_)
        self.goal_.p.z = saturate(self.goal_.p.z, self.zmin_, self.zmax_)

        self.goal_.header.stamp = rospy.Time.now()  # Set current time

        # Goals should only be published here because this is the only place where we
        # apply safety bounds. Exceptions: when killing the drone and when clicking END at init pos traj
        # self.pub_wind_.publish(self.wind_)
        self.pub_goal_.publish(self.goal_)
    
    def generate_trajectory(self):
        print("Generating trajectories...")
        if self.traj_type == 'point':
            points = [
                np.array([1, 1, 1]),
            ]

            point_traj = Point(points)
            all_goals = point_traj.generate_all_trajectories()

        elif self.traj_type == 'circle':
            radius = 2.0
            center_x = 0.0
            center_y = 0.0
            alt = 1.8

            circle_traj = Circle(self.T, self.dt_, radius, center_x, center_y, alt)
            all_goals = circle_traj.generate_all_trajectories()

        elif self.traj_type == 'figure_eight':
            a = 3.0
            b = 1.5
            center_x = 0.0
            center_y = 0.0
            alt = 1.8

            figure_eight_traj = FigureEight(self.T, self.dt_, a, b, center_x, center_y, alt)
            all_goals = figure_eight_traj.generate_all_trajectories()

        else:
            spline_traj = Spline(self.num_traj, self.T, self.dt_, self.key, self.xmin_, self.ymin_, self.zmin_, self.xmax_, self.ymax_, self.zmax_)
            all_goals = spline_traj.generate_all_trajectories()

        print("Finished generating trajectories...")
        return all_goals

    def generate_wind(self): 
        wind_sim = WindSim(self.key, self.num_traj, self.T, self.dt_, self.wind_type)
        all_winds = wind_sim.generate_all_winds()
        
        return all_winds

    def reset_goal(self):
        # Creating a new goal message should already set this correctly, but just in case
        # Exception: yaw would be 0 instead of current yaw
        self.goal_.p.x, self.goal_.p.y, self.goal_.p.z = self.pose_.position.x, self.pose_.position.y, self.pose_.position.z
        self.goal_.v.x, self.goal_.v.y, self.goal_.v.z = self.vel_.linear.x, self.vel_.linear.y, self.vel_.linear.z
        self.goal_.a.x, self.goal_.a.y, self.goal_.a.z = 0, 0, 0
        self.goal_.j.x, self.goal_.j.y, self.goal_.j.z = 0, 0, 0
        #self.goal_.s.x, self.goal_.s.y, self.goal_.s.z = 0, 0, 0
        self.goal_.psi = 0
        self.goal_.dpsi = 0
        # self.goal_.power = False
        # reset_xy_int and  reset_z_int are not used
        self.goal_.mode_xy = Goal.MODE_POSITION_CONTROL
        self.goal_.mode_z = Goal.MODE_POSITION_CONTROL
    
    def reset_wind(self):
        self.wind_.w_nominal.x, self.wind_.w_nominal.y, self.wind_.w_nominal.z = 0, 0, 0
        self.wind_.w_gust.x, self.wind_.w_gust.y, self.wind_.w_gust.z = 0, 0, 0
    
    def record_data(self):
        if self.record:
            goal_index = self.index
            traj_index = self.pub_index_

            self.q[goal_index, traj_index] = np.array([self.pose_.position.x, self.pose_.position.y, self.pose_.position.z])
            self.dq[goal_index, traj_index] = np.array([self.vel_.linear.x, self.vel_.linear.y, self.vel_.linear.z])

            self.u[goal_index, traj_index] = np.array([self.att_cmd_.F_W.x, self.att_cmd_.F_W.y, self.att_cmd_.F_W.z])

            self.quat[goal_index, traj_index] = np.array([self.pose_.orientation.w, self.pose_.orientation.x, self.pose_.orientation.y, self.pose_.orientation.z])
            self.omega[goal_index, traj_index] = np.array([self.vel_.angular.x, self.vel_.angular.y, self.vel_.angular.z])

            self.r[goal_index, traj_index] = np.array([self.goal_.p.x, self.goal_.p.y, self.goal_.p.z])
            self.dr[goal_index, traj_index] = np.array([self.goal_.v.x, self.goal_.v.y, self.goal_.v.z])
    
    # def publish_data(self):
    #     print("Writing data...")

    #     data = {
    #         'seed': self.seed, 'prng_key': self.key,
    #         't': self.t, 'q': self.q, 'dq': self.dq,
    #         'u': self.u, 'r': self.r, 'dr': self.dr,
    #         'quat': self.quat, 'omega': self.omega,
    #         't_knots': self.t_knots, 'r_knots': self.r_knots,
    #         'w': self.w, 'w_min': self.w_min, 'w_max': self.w_max,
    #         'beta_params': (self.a, self.b),
    #     }

    #     current_time = datetime.datetime.now()
    #     timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    #     rospack = rospkg.RosPack()
    #     package_path = rospack.get_path('trajectory_generator_python')
    #     file_path = package_path + f'/data/{timestamp}_traj{self.num_traj}_seed{self.seed}.pkl'

    #     with open(file_path, 'wb+') as file:
    #         pickle.dump(data, file)

def main():
    # Initialize the ROS node with the default name 'my_node_name' (will be overwritten by launch file)
    rospy.init_node('my_node_name')
    TrajectoryGenerator()
    # Spin to keep the node alive and process callbacks
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
