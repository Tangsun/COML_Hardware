#!/usr/bin/env python

import numpy as np

def wrapToPi(x):
    """Wrap To Pi
    Enforce x (scalar or np array) to be btwn [-pi, pi]
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrapTo2Pi(x):
    """Wrap To 2Pi
    Enforce x (scalar or np array) to be btwn [0, 2*pi]
    """
    return x % (2 * np.pi)


def clamp_indicate(val, low, high):
    clamped = False
    if val > high:
        val = high
        clamped = True
    if val < low:
        val = low
        clamped = True
    return val, clamped


def clamp(val, low, high):
    val, clamped = clamp_indicate(val, low, high)
    return val


def rateLimit(dt, lRateLim, uRateLim, v0, v1):
    """Rate Limit
    Saturate the current valule of a signal by limiting its rate of change,
    given its current (desired) value, last value, and the timestep.
    """
    if v1 > (v0 + uRateLim * dt):
        v1 = v0 + uRateLim * dt
    elif v1 < (v0 + lRateLim * dt):
        v1 = v0 + lRateLim * dt
    return v1


def expo(x, e):
    """Uses a cubic polynomial for 'expo'
    See http://www.mimirgames.com/articles/games/joystick-input-and-using-deadbands/
    """
    return e*x**3 + (1-e)*x


class Trajectory:
    """
    Creates a trajectory up to constant jerk for a straight line path.

    See https://jwdinius.github.io/blog/2018/eta3traj
    """
    def __init__(self, ps, pe, max_vel=0.35, v0=0.0, a0=0.0, max_accel=0.5, max_jerk=1.0):
       # ensure that all inputs obey the assumptions of the model
        assert max_vel > 0 and v0 >= 0 and a0 >= 0 and max_accel > 0 and max_jerk > 0 \
            and a0 <= max_accel and v0 <= max_vel

        self.v_max_user = max_vel
        self.v0 = v0
        self.a0 = a0
        self.a_max = max_accel
        self.j_max = max_jerk
        self.ps = ps
        self.pe = pe

        # calculate total length of input path
        self.stotal = np.abs(pe-ps)

        ## compute velocity profile on top of the path
        self.velocity_profile()

    def velocity_profile(self):
        '''                   /~~~~~----------------~~~~~\
                             /                            \
                            /                              \
                           /                                \
                          /                                  \
        (v=v0, a=a0) ~~~~~                                    \
                                                               \
                                                                \~~~~~ (vf=0, af=0)
                     pos.|pos.|neg.|   cruise at    |neg.| neg. |neg.
                     max |max.|max.|     max.       |max.| max. |max.
                     jerk|acc.|jerk|    velocity    |jerk| acc. |jerk
            index     0    1    2      3 (optional)   4     5     6
        '''
        ## First, calculate maximum attainable velocity
        # Segment 1
        dt1 = (self.a_max - self.a0) / self.j_max
        ds1 = self.v0 * dt1 + self.a0 * dt1**2 / 2. + self.j_max * dt1**3 / 6.
        v1 = self.v0 + self.a0 * dt1 + self.j_max * dt1**2 / 2.
        # Segment 7
        dt7 = self.a_max / self.j_max
        ds7 = self.j_max * dt7**3 / 6.
        # solve for the maximum achievable velocity based on the kinematic limits imposed by max_accel and max_jerk
        # this leads to a quadratic equation in v_max: a*v_max**2 + b*v_max + c = 0
        a = 1 / self.a_max
        b = self.a_max / self.j_max
        c = ds1 + ds7 - self.stotal - (5. * self.a_max**3) / (24. * self.j_max**2) \
            - v1**2 / (2. * self.a_max)
        v_max = ( -b + np.sqrt(b**2 - 4. * a * c) ) / (2. * a)

        ## Then, subject this maximum attainable velocity by the user supplied maximum velocity.
        self.v_max = self.v_max_user if v_max > self.v_max_user else v_max

        # setup arrays to store values at END of trajectory sections
        self.dt = np.zeros((7,))
        self.ds = np.zeros((7,))
        self.v = np.zeros((7,))

        # Segment 1: max jerk up to max acceleration
        i = 0
        self.dt[0] = dt1
        self.ds[0] = ds1
        self.v[0] = v1

        # Prelude: Segment 3
        dt3 = self.a_max/self.j_max
        dv3 = self.j_max*dt3**2 / 2.

        # Segment 2: accelerate at max_accel
        i = 1
        self.v[i] = self.v_max - dv3
        self.dt[i] = (self.v[i] - self.v[i-1]) / self.a_max
        self.ds[i] = self.v[i-1] * self.dt[i] + self.a_max * self.dt[i]**2 / 2.

        # Segment 3: decrease acceleration (down to 0) until max speed is hit
        i = 2
        self.dt[i] = dt3
        self.ds[i] = self.v[i-1] * self.dt[i] + self.a_max * self.dt[i]**2 / 2. \
            - self.j_max * self.dt[i]**3 / 6.
        self.v[i] = self.v[i-1] + self.a_max * self.dt[i] \
            - self.j_max * self.dt[i]**2 / 2.

        # as a check, the velocity at the end of the section should be self.v_max
        if not np.isclose(self.v[i], self.v_max):
            raise MaxVelocityNotReached(self.v[i], self.v_max)

        # Segment 4: cruise; save for last

        # Segment 5: apply min jerk until min acceleration is hit
        i = 4
        self.dt[i] = self.a_max / self.j_max
        self.ds[i] = self.v_max * self.dt[i] - self.j_max * self.dt[i]**3 / 6.
        self.v[i] = self.v_max - self.j_max * self.dt[i]**2 / 2.

        # Prelude: Segment 7
        dt7 = self.a_max/self.j_max
        dv7 = -self.j_max*dt7**2 / 2.

        # Segment 6: continue deceleration at max rate
        i = 5
        self.v[i] = -dv7
        self.dt[i] = (self.v[i] - self.v[i-1]) / -self.a_max
        self.ds[i] = self.v[i-1] * self.dt[i] - self.a_max * self.dt[i]**2 / 2.

        # Segment 7: max jerk to get to zero velocity and zero acceleration simultaneously
        i = 6
        self.dt[i] = dt7
        self.ds[i] = self.v[i-1] * self.dt[i] - self.a_max * self.dt[i]**2 / 2. \
            + self.j_max * self.dt[i]**3 / 6.
        self.v[i] = self.v[i-1] - self.j_max * self.dt[i]**2 / 2.

        try:
            assert np.isclose(self.v[i], 0)
        except AssertionError as e:
            print('The final velocity {} is not zero'.format(self.v[i]))
            raise e

        # Finale: Segment 4
        if self.ds.sum() < self.stotal:
            i = 3
            # the length of the cruise section is whatever length hasn't already been accounted for
            # NOTE: the total array self.ds is summed because the entry for the cruise segment is
            # initialized to 0!
            self.ds[i] = self.stotal - self.ds.sum()
            self.dt[i] = self.ds[i] / self.v_max
            self.v[i] = self.v_max

        # make sure that all of the times are positive, otherwise the kinematic limits
        # chosen cannot be enforced on the path
        assert(np.all(self.dt >= 0))

    def at_time(self, time):
        assert(time >= 0)

        # compute velocity at time
        if time <= self.dt[0]:
            p = self.v0 * time + self.a0 * time**2 / 2. + self.j_max * time**3 / 6
            v = self.v0 + self.a0 * time + self.j_max * time**2 / 2.
            a = self.a0 + self.j_max * time
            j = self.j_max

        elif time <= self.dt[:2].sum():
            delta_t = time - self.dt[0]
            p = self.ds[0] + self.v[0] * delta_t + self.a_max * delta_t**2 / 2.
            v = self.v[0] + self.a_max * delta_t
            a = self.a_max
            j = 0.

        elif time <= self.dt[:3].sum():
            delta_t = time - self.dt[:2].sum()
            p = self.ds[:2].sum() + self.v[1] * delta_t + self.a_max * delta_t**2 /2. \
                - self.j_max * delta_t**3 / 6.
            v = self.v[1] + self.a_max * delta_t - self.j_max * delta_t**2 / 2.
            a = self.a_max - self.j_max * delta_t
            j = -self.j_max

        elif time <= self.dt[:4].sum():
            delta_t = time - self.dt[:3].sum()
            p = self.ds[:3].sum() + self.v[3] * delta_t
            v = self.v[3]
            a = 0.
            j = 0.

        elif time <= self.dt[:5].sum():
            delta_t = time - self.dt[:4].sum()
            p = self.ds[:4].sum() + self.v_max * delta_t - self.j_max * delta_t**3 / 6.
            v = self.v_max - self.j_max * delta_t**2 / 2.
            a = -self.j_max * delta_t
            j = -self.j_max

        elif time <= self.dt[:-1].sum():
            delta_t = time - self.dt[:5].sum()
            p = self.ds[:5].sum() + self.v[4] * delta_t - self.a_max * delta_t**2 / 2.
            v = self.v[4] - self.a_max * delta_t
            a = -self.a_max
            j = 0.

        elif time < self.dt.sum():
            delta_t = time - self.dt[:-1].sum()
            p = self.ds[:-1].sum() + self.v[5] * delta_t - self.a_max * delta_t**2 / 2. \
                + self.j_max * delta_t**3 / 6.
            v = self.v[5] - self.a_max * delta_t + self.j_max * delta_t**2 / 2.
            a = -self.a_max + self.j_max*delta_t
            j = self.j_max

        else:
            v = 0.
            p = self.stotal
            a = 0.
            j = 0.

        # start path at user-specified starting point
        p = self.ps + p

        return (p, v, a, j)
