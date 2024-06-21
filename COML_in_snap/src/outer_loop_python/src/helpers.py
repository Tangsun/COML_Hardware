import numpy as np
from geometry_msgs.msg import Point, Quaternion, Vector3

# convert from ROS message to array
def point_msg_to_array(point_msg):
    return np.array([point_msg.x, point_msg.y, point_msg.z])

def vector_msg_to_array(vector_msg):
    return np.array([vector_msg.x, vector_msg.y, vector_msg.z])

def quaternion_msg_to_quaternion(quaternion_msg):
    return np.array([quaternion_msg.w, quaternion_msg.x, quaternion_msg.y, quaternion_msg.z])

# convert from array to ROS message
def point_array_to_msg(point_array):
    return Point(x=point_array[0], y=point_array[1], z=point_array[2])

def vector_array_to_msg(vector_array):
    return Vector3(x=vector_array[0], y=vector_array[1], z=vector_array[2])

def quaternion_array_to_msg(quaternion_array):
    return Quaternion(w=quaternion_array[0], x=quaternion_array[1], y=quaternion_array[2], z=quaternion_array[3])

def quaternion_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
                     
def get_rpy(quaternion):
    """Computes the roll, pitch, and yaw from a normalized quaternion array."""
    w, x, y, z = quaternion
    # Normalize the quaternion to avoid errors in computation
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return Vector3(roll, pitch, yaw)
