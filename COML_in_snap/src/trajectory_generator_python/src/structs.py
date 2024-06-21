from enum import Enum

class FlightMode(Enum):
    GROUND = 0
    TAKING_OFF = 1
    HOVERING = 2
    INIT_POS_TRAJ = 3
    TRAJ_FOLLOWING = 4
    LANDING = 5
    INIT_POS = 6
