# import jax.numpy as jnp
import numpy as np
import os
import pickle
import rospkg

from dynamics import prior
from structs import AttCmdClass, ControlLogClass, GoalClass
from helpers import quaternion_multiply
from utils import params_to_posdef, quaternion_to_rotation_matrix

def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)

class IntegratorClass:
    def __init__(self):
        self.value_ = 0

    def increment(self, inc, dt):
        self.value_ += inc * dt

    def reset(self):
        self.value_ = 0

    def value(self):
        return self.value_

class OuterLoop:
    def __init__(self, params, state0, goal0, controller='coml'):
        self.controller = controller

        if self.controller == 'coml':
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('outer_loop_python')

            trial_name = 'reg_P_1_reg_Kr_1e-3'
            filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=1.0000.pkl'

            # trial_name = 'reg_P_2e-3_reg_k_R_2e-3_k_R_z_1'
            # filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=0.0020.pkl'

            # trial_name = 'reg_P_1e-1_reg_k_R_0'
            # filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=0.1000.pkl'

            model_dir = f'{package_path}/models/{trial_name}'
            model_pkl_loc = os.path.join(model_dir, filename)
            with open(model_pkl_loc, 'rb') as f:
                train_results = pickle.load(f)
            
            print('COML model loaded!')
            
            self.pnorm = convert_p_qbar(train_results['pnorm'])
            self.W = train_results['model']['W']
            self.b = train_results['model']['b']
            self.Λ = params_to_posdef(train_results['controller']['Λ'])
            self.K = params_to_posdef(train_results['controller']['K'])
            self.P = params_to_posdef(train_results['controller']['P'])
        
        self.params_ = params
        self.GRAVITY = np.array([0, 0, -9.80665])

        self.Ix_ = IntegratorClass()
        self.Iy_ = IntegratorClass()
        self.Iz_ = IntegratorClass()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0

        self.mode_xy_last_ = GoalClass.Mode.POS_CTRL
        self.mode_z_last_ = GoalClass.Mode.POS_CTRL
        
        self.reset(state0, goal0)

    def reset(self, state0, goal0):
        if self.controller == 'coml':
            # Assume starting position, velocity, attitude, and angular velocity of 0
            q0 = state0.p
            dq0 = state0.v
            R_flatten0 = quaternion_to_rotation_matrix(state0.q).flatten()
            Omega0 = state0.w
            r0 = goal0.p
            dr0 = goal0.v
            
            self.dA_prev, y0 = self.adaptation_law(q0, dq0, R_flatten0, Omega0, r0, dr0)
            self.pA_prev = np.zeros((q0.size, y0.size))
            
        self.Ix_.reset()
        self.Iy_.reset()
        self.Iz_.reset()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0
    
    
    def update_log(self, state):
        self.log_ = ControlLogClass()
        self.log_.p = state.p
        self.log_.v = state.v
        self.log_.q = state.q
        self.log_.w = state.w

    def compute_attitude_command(self, t, state, goal):
        dt = 1e-2 if self.t_last_ == 0 else t - self.t_last_

        if dt > 0:
            self.t_last_ = t
        else:
            print("Warning: non-positive dt:", dt, "[s].")
        
        if self.controller == 'coml':
            qn = 1.1 + self.pnorm**2

            # Integrate adaptation law via trapezoidal rule
            R_flatten = quaternion_to_rotation_matrix(state.q).flatten()
            dA, y = self.adaptation_law(state.p, state.v, R_flatten, state.w, goal.p, goal.v)
            pA = self.pA_prev + (dt)*(self.dA_prev + dA)/2
            P = self.P
            A = (np.maximum(np.abs(pA), 1e-6 * np.ones_like(pA))**(qn-1) * np.sign(pA) * (np.ones_like(pA) - np.isclose(pA, 0, atol=1e-6)) ) @ P

            f_hat = A @ y

            # Log adaptation values
            self.log_.P_norm = np.linalg.norm(P)
            self.log_.A_norm = np.linalg.norm(A)
            self.log_.y_norm = np.linalg.norm(y)
            self.log_.f_hat = f_hat

            # Update prev values
            self.pA_prev = pA
            self.dA_prev = dA
        else:
            f_hat = 0

        F_W = self.get_force(dt, state, goal, f_hat)
        q_ref = self.get_attitude(state, goal, F_W)
        w_ref = self.get_rates(dt, state, goal, F_W, self.log_.a_fb, q_ref)

        cmd = AttCmdClass()
        # print('q_ref:', q_ref)
        cmd.q = q_ref
        cmd.w = w_ref
        cmd.F_W = F_W

        return cmd

    def adaptation_law(self, q, dq, R_flatten, Omega, r, dr):
        # Regressor features
        y = np.concatenate((q, dq, R_flatten, Omega))
        for W, b in zip(self.W, self.b):
            y = np.tanh(W@y + b)

        # Auxiliary signals
        Λ, P = self.Λ, self.P
        e, de = q - r, dq - dr
        s = de + Λ@e

        dA = np.outer(s, y) @ P
        return dA, y

    def get_force(self, dt, state, goal, f_hat):
        if self.controller == 'coml':
            # Auxiliary signals
            Λ, K = self.Λ, self.K

            e, edot = state.p - goal.p, state.v - goal.v
            s = edot + Λ@e
            v, dv = goal.v - Λ@e, goal.a - Λ@edot

            # Control input and adaptation law
            H, C, g, B = prior(state.p, state.v)
            τ = H@dv + C@v + g - f_hat - K@s
            F_W = np.linalg.solve(B, τ)
        else:
            e = goal.p - state.p
            edot = goal.v - state.v

            # Saturate error so it isn't too much for control gains
            e = np.minimum(np.maximum(e, -self.params_.maxPosErr), self.params_.maxPosErr)
            edot = np.minimum(np.maximum(edot, -self.params_.maxVelErr), self.params_.maxVelErr)

            # Manipulate error signals based on selected flight mode
            # Reset integrators on mode change
            if goal.mode_xy != self.mode_xy_last_:
                self.Ix_.reset()
                self.Iy_.reset()
                self.mode_xy_last_ = goal.mode_xy

            if goal.mode_z != self.mode_z_last_:
                self.Iz_.reset()
                self.mode_z_last_ = goal.mode_z

            # Check which control mode to use for x-y
            if goal.mode_xy == GoalClass.Mode.POS_CTRL:
                self.Ix_.increment(e[0], dt)
                self.Iy_.increment(e[1], dt)
            elif goal.mode_xy == GoalClass.Mode.VEL_CTRL:
                # Do not worry about position error---only vel error
                e[0] = e[1] = 0
            elif goal.mode_xy == GoalClass.Mode.ACC_CTRL:
                # Do not generate feedback accel---only control on goal accel
                e[0] = e[1] = 0
                edot[0] = edot[1] = 0

            # Check which control mode to use for z
            if goal.mode_z == GoalClass.Mode.POS_CTRL:
                self.Iz_.increment(e[2], dt)
            elif goal.mode_z == GoalClass.Mode.VEL_CTRL:
                # Do not worry about position error---only vel error
                e[2] = 0
            elif goal.mode_z == GoalClass.Mode.ACC_CTRL:
                # Do not generate feedback accel---only control on goal accel
                e[2] = 0
                edot[2] = 0

            # Compute feedback acceleration via PID, eq (2.9)
            eint = np.array([self.Ix_.value(), self.Iy_.value(), self.Iz_.value()])
            a_fb = np.multiply(self.params_.Kp, e) + np.multiply(self.params_.Ki, eint) + np.multiply(self.params_.Kd, edot)

            # Compute total desired force (expressed in world frame), eq (2.12)
            F_W = self.params_.mass * (goal.a + a_fb - self.GRAVITY)

        # Log control signals for debugging and inspection
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = e
        # self.log_.p_err_int = eint
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = edot
        self.log_.a_ff = goal.a
        # self.log_.a_fb = a_fb
        self.log_.F_W = F_W

        # Return total desired force expressed in world frame
        return F_W

    def get_attitude(self, state, goal, F_W):
        xi = F_W / self.params_.mass  # Eq. 26
        abc = xi / np.linalg.norm(xi)  # Eq. 19

        a, b, c = abc
        psi = goal.psi

        invsqrt21pc = 1 / np.sqrt(2 * (1 + c))

        quaternion0 = np.array([invsqrt21pc*(1+c), invsqrt21pc*(-b), invsqrt21pc*a, 0.0])
        quaternion1 = np.array([np.cos(psi/2.), 0.0, 0.0, np.sin(psi/2.)])

        # Construct the quaternion
        q_ref = quaternion_multiply(quaternion0, quaternion1)

        # Normalize quaternion
        q_ref = q_ref / np.linalg.norm(q_ref)

        # TODO: Implement the second fibration for the whole SO(3)
        # See Eq. 22, 23, and 24

        # Log control signals for debugging and inspection
        self.log_.q = state.q
        self.log_.q_ref = q_ref

        return q_ref

    def get_rates(self, dt, state, goal, F_W, a_fb, q_ref):
        # Generate feedback jerk by numerical derivative of feedback accel
        j_fb = np.zeros(3)
        if dt > 0:
            # Numeric derivative
            j_fb = (a_fb - self.a_fb_last_) / dt

            # Low-pass filter differentiation
            tau = 0.1
            alpha = dt / (tau + dt)
            j_fb = alpha * j_fb + (1 - alpha) * self.j_fb_last_
        else:
            # Re-use last value
            j_fb = self.j_fb_last_

        # Save for the next time
        self.a_fb_last_ = a_fb
        self.j_fb_last_ = j_fb

        # Construct angular rates consistent with trajectory dynamics
        Fdot_W = goal.j + j_fb
        xi = F_W / self.params_.mass  # Eq. 26
        abc = xi / np.linalg.norm(xi)  # Eq. 19
        xi_dot = Fdot_W / self.params_.mass
        I = np.eye(3)
        norm_xi = np.linalg.norm(xi)

        abcdot = ((norm_xi**2 * I - np.outer(xi, xi)) / norm_xi**3) @ xi_dot  # Eq. 20

        # Assert abc' * abcdot should be approximately 0.0
        assert np.allclose(np.dot(abc, abcdot), 0.0)

        a, b, c = abc
        adot, bdot, cdot = abcdot
        psi, psidot = goal.psi, goal.dpsi

        rates = np.zeros(3)
        rates[0] = np.sin(psi) * adot - np.cos(psi) * bdot - (a * np.sin(psi) - b * np.cos(psi)) * (cdot / (c + 1))
        rates[1] = np.cos(psi) * adot + np.sin(psi) * bdot - (a * np.cos(psi) + b * np.sin(psi)) * (cdot / (c + 1))
        rates[2] = (b * adot - a * bdot) / (1 + c) + psidot

        # Log control signals for debugging and inspection
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = goal.p - state.p
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = goal.v - state.v
        self.log_.a_ff = goal.j
        self.log_.a_fb = j_fb
        self.log_.F_W = F_W
        self.log_.j_ff = goal.j
        self.log_.j_fb = j_fb
        self.log_.w = state.w
        self.log_.w_ref = rates

        return rates

    def get_log(self):
        return self.log_
