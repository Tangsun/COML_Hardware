import math
from snapstack_msgs.msg import Goal
import numpy as np
import rospkg
import subprocess

def quat2yaw(q) -> float:
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                    1 - 2 * (q.y * q.y + q.z * q.z))
    return yaw

def saturate(val, low, high):
        return max(min(val, high), low)

def wrap(val):
    while val > math.pi:
        val -= 2.0 * math.pi
    while val < -math.pi:
        val += 2.0 * math.pi
    return val

def simpleInterpolation(current, dest_pos, dest_yaw, vel, vel_yaw,
                            dist_thresh, yaw_thresh, dt, finished):
    goal = Goal()  # Assuming Goal is a class representing snapstack_msgs::Goal

    Dx = dest_pos.p.x - current.p.x
    Dy = dest_pos.p.y - current.p.y
    Dz = dest_pos.p.z - current.p.z
    dist = math.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)

    delta_yaw = dest_yaw - current.psi
    delta_yaw = wrap(delta_yaw)

    dist_far = dist > dist_thresh
    yaw_far = abs(delta_yaw) > yaw_thresh
    finished = not dist_far and not yaw_far

    accel_for_vel = 0.1

    # goal.p.z = dest_pos.p.z

    if dist_far:
        c = Dx / dist
        s = Dy / dist
        v = Dz / dist
        goal.p.x = current.p.x + c * vel * dt
        goal.p.y = current.p.y + s * vel * dt
        goal.p.z = current.p.z + v * vel * dt

        goal.v.x = min(current.v.x + accel_for_vel * dt, c * vel)
        goal.v.y = min(current.v.y + accel_for_vel * dt, s * vel)
        goal.v.z = min(current.v.z + accel_for_vel * dt, v * vel)
    else:
        goal.p.x = dest_pos.p.x
        goal.p.y = dest_pos.p.y
        goal.p.z = dest_pos.p.z

        goal.v.x = max(0.0, current.v.x - accel_for_vel * dt)
        goal.v.y = max(0.0, current.v.y - accel_for_vel * dt)
        goal.v.z = max(0.0, current.v.z - accel_for_vel * dt)

    if yaw_far:
        sgn = 1 if delta_yaw >= 0 else -1
        vel_yaw = sgn * vel_yaw
        goal.psi = current.psi + vel_yaw * dt
        goal.dpsi = vel_yaw
    else:
        goal.psi = dest_yaw
        goal.dpsi = 0

    goal.power = True

    return goal, finished

def start_rosbag_recording(topic_name):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('outer_loop_python')
    # Define the command to start rosbag recording
    command = ['rosbag', 'record', '-q', '-o', f'{package_path}/rosbags/real/', topic_name]
    # Start recording
    rosbag_proc = subprocess.Popen(command)
    print('Started rosbag recording!')
    return rosbag_proc

def stop_rosbag_recording(rosbag_proc):
    # Terminate the rosbag recording process
    rosbag_proc.send_signal(subprocess.signal.SIGINT)
    print('Stopped rosbag recording!')