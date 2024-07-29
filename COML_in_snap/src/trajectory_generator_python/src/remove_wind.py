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

if __name__ == "__main__":
    print("entered script")

    with open('./src/trajectory_generator_python/data/2024-07-12_14-21-20_traj50_seed2.pkl', 'rb') as file:
        raw = pickle.load(file)

    print("Loaded in pickle")

    new_data = {
            'seed': raw['seed'], 'prng_key': raw['prng_key'],
            't': raw['t'], 'q': raw['q'], 'dq': raw['dq'],
            'u': raw['u'],
            'r': raw['r'], 'dr': raw['dr'],
            'quat': raw['quat'], 'omega': raw['omega'],
            't_knots': raw['t_knots'], 'r_knots': raw['r_knots'],
            'w_min': raw['w_min'], 'w_max': raw['w_max'],
            'beta_params': raw['beta_params'],
        }
    
    print("copied data to new struct")
    
    with open('./src/trajectory_generator_python/data/2024-06-26_12-41-43_traj50_seed2_no_wind.pkl', 'wb') as file:
        raw = pickle.dump(new_data, file)

    print("wrote new struct")