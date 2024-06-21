from enum import Enum
import numpy as np

class AttCmdClass:
    def __init__(self):
        self.q = np.array([1, 0, 0, 0]) # w, x, y, z
        self.w = np.zeros(3)
        self.F_W = np.zeros(3)

class ParametersClass:
    def __init__(self):
        self.mass = 0.0
        self.Kp = np.zeros(3)
        self.Ki = np.zeros(3)
        self.Kd = np.zeros(3)
        self.maxPosErr = np.zeros(3)
        self.maxVelErr = np.zeros(3)

class StateClass:
    def __init__(self):
        self.t = -1
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1, 0, 0, 0]) # w, x, y, z
        self.w = np.zeros(3)

class GoalClass:
    class Mode(Enum):
        POS_CTRL = 0
        VEL_CTRL = 1
        ACC_CTRL = 2

    def __init__(self):
        self.mode_xy = self.Mode.POS_CTRL
        self.mode_z = self.Mode.POS_CTRL
        self.t = -1
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.j = np.zeros(3)
        self.psi = 0
        self.dpsi = 0

class ControlLogClass:
    def __init__(self):
        self.p = np.zeros(3)
        self.p_ref = np.zeros(3)
        self.p_err = np.zeros(3)
        self.p_err_int = np.zeros(3)
        self.v = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.v_err = np.zeros(3)
        self.a_ff = np.zeros(3)
        self.a_fb = np.zeros(3)
        self.j_ff = np.zeros(3)
        self.j_fb = np.zeros(3)
        self.q = np.array([1, 0, 0, 0]) # w, x, y, z
        self.q_ref = np.array([1, 0, 0, 0]) # w, x, y, z
        self.w = np.zeros(3)
        self.w_ref = np.zeros(3)
        self.F_W = np.zeros(3)  # total desired force [N], expr in world frame
        self.mode_xy = GoalClass.Mode.POS_CTRL
        self.mode_z = GoalClass.Mode.POS_CTRL

        self.P_norm = 0
        self.A_norm = 0
        self.y_norm = 0
        self.f_hat = np.zeros(3)

class ModeClass(Enum):
    Preflight = 0
    SpinningUp = 1
    Flying = 2
    EmergencyStop = 3