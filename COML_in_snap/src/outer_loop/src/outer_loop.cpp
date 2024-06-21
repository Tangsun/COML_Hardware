/**
 * @file outer_loop.cpp
 * @brief Outer loop trajectory tracking snap-stack controller
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @author Jesus Tordesillas Torres <jotrde@mit.edu>
 * @date 20 August 2020
 *
 * @note We use ROS standard frames: inertial ENU with body flu
 */

#include <outer_loop/outer_loop.h>

#include <iostream>

namespace acl {
namespace outer_loop {
OuterLoop::OuterLoop(const Parameters& params)
: params_(params)
{
  reset();
}

// ----------------------------------------------------------------------------

void OuterLoop::reset()
{
  Ix_.reset();
  Iy_.reset();
  Iz_.reset();
  log_ = ControlLog();
  a_fb_last_ = Eigen::Vector3d::Zero();
  j_fb_last_ = Eigen::Vector3d::Zero();
  t_last_ = 0;
}

// ----------------------------------------------------------------------------

void OuterLoop::updateLog(const State& state)
{
  log_ = ControlLog();
  log_.p = state.p;
  log_.v = state.v;
  log_.q = state.q;
  log_.w = state.w;
}

// ----------------------------------------------------------------------------

void OuterLoop::setHorizontalPositionPidParams(const double kp, const double ki, const double kd){
  // P gain on x-y (Inertial frame)
  params_.Kp(0) = kp;
  params_.Kp(1) = kp;

  // I gain on x-y (Inertial frame)
  params_.Ki(0) = ki;
  params_.Ki(1) = ki;

  // D-gain on x-y (Inertial frame)
  params_.Kd(0) = kd;
  params_.Kd(1) = kd;
}

// ----------------------------------------------------------------------------

void OuterLoop::setAltitudePidParams(const double kp, const double ki, const double kd){
  // P gain on z (Inertial frame)
  params_.Kp(2) = kp;

  // I gain on z (Inertial frame)
  params_.Ki(2) = ki;

  // D-gain on z (Inertial frame)
  params_.Kd(2) = kd;
}

// ----------------------------------------------------------------------------

AttCmd OuterLoop::computeAttitudeCommand(double t, const State& state,
                                         const Goal& goal)
{
  // if first loop, set dt to something reasonable
  const double dt = (t_last_ == 0) ? 1e-2 : t - t_last_;
  if (dt > 0) {
    t_last_ = t;
  } else {
    std::cout << "Warning: non-positive dt: " << dt << " [s]." << std::endl;
  }

  // compute desired force (expr in world frame) via PID
  const Eigen::Vector3d F_W = getForce(dt, state, goal);

  // determine des attitude from des total force vector (Markley min quat)
  const Eigen::Quaterniond q_ref = getAttitude(state, goal, F_W);

  // construct dynamically consistent angular rates
  const Eigen::Vector3d w_ref = getRates(dt, state, goal, F_W, log_.a_fb, q_ref);

  // Build attitude command
  AttCmd cmd;
  cmd.q = q_ref;
  cmd.w = w_ref;
  cmd.F_W = F_W;

  return cmd;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

Eigen::Vector3d OuterLoop::getForce(double dt, const State& state, const Goal& goal)
{
  // For referenced equations, see Lopez SM'16
  // (https://dspace.mit.edu/handle/1721.1/107052).
  // See also Cutler and How, "Actuator Constrained Traj Gen..., 2012"
  // (http://acl.mit.edu/papers/2012-uber-compressed.pdf).

  // Calculate error, eq (2.10) and (2.11)
  //

  Eigen::Vector3d e = goal.p - state.p;
  Eigen::Vector3d edot = goal.v - state.v;

  // saturate error so it isn't too much for control gains
  e = e.cwiseMin(params_.maxPosErr).cwiseMax(-params_.maxPosErr);
  edot = edot.cwiseMin(params_.maxVelErr).cwiseMax(-params_.maxVelErr);

  //
  // Manipulate error signals based on selected flight mode
  //

  // reset integrators on mode change
  if (goal.mode_xy != mode_xy_last_) {
    Ix_.reset();
    Iy_.reset();
    mode_xy_last_ = goal.mode_xy;
  }

  if (goal.mode_z != mode_z_last_) {
    Iz_.reset();
    mode_z_last_ = goal.mode_z;
  }

  // check which control mode to use
  if (goal.mode_xy == Goal::Mode::POS_CTRL) {
    Ix_.increment(e.x(), dt);
    Iy_.increment(e.y(), dt);
  } else if (goal.mode_xy == Goal::Mode::VEL_CTRL) {
    // do not worry about position error---only vel error
    e.x() = e.y() = 0;
  } else if (goal.mode_xy == Goal::Mode::ACC_CTRL) {
    // do not generate feedback accel---only control on goal accel
    e.x() = e.y() = 0;
    edot.x() = edot.y() = 0;
  }

  if (goal.mode_z == Goal::Mode::POS_CTRL) {
    Iz_.increment(e.z(), dt);
  } else if (goal.mode_z == Goal::Mode::VEL_CTRL) {
    // do not worry about position error---only vel error
    e.z() = 0;
  } else if (goal.mode_z == Goal::Mode::ACC_CTRL) {
    // do not generate feedback accel---only control on goal accel
    e.z() = 0;
    edot.z() = 0;
  }

  //
  // Compute feedback acceleration via PID, eq (2.9)
  //

  const Eigen::Vector3d eint(Ix_.value(), Iy_.value(), Iz_.value());

  const Eigen::Vector3d a_fb = params_.Kp.cwiseProduct(e)
                               + params_.Ki.cwiseProduct(eint)
                               + params_.Kd.cwiseProduct(edot);

  //
  // Compute total desired force (expressed in world frame), eq (2.12)
  //

  const Eigen::Vector3d F_W = params_.mass * (goal.a + a_fb - GRAVITY);

  //
  // Log control signals for debugging and inspection
  //

  log_.p = state.p;
  log_.p_ref = goal.p;
  log_.p_err = e;
  log_.p_err_int = eint;
  log_.v = state.v;
  log_.v_ref = goal.v;
  log_.v_err = edot;
  log_.a_ff = goal.a;
  log_.a_fb = a_fb;
  log_.F_W = F_W;

  // return total desired force expr in world frame
  return F_W;
}

// ----------------------------------------------------------------------------

Eigen::Quaterniond OuterLoop::getAttitude(const State& state, const Goal& goal,
                                          const Eigen::Vector3d& F_W)
{
  // For referenced equations, see
  // "Control of Quadrotors Using the Hopf Fibration on SO(3)",
  // https://link.springer.com/chapter/10.1007/978-3-030-28619-4_20

  const Eigen::Vector3d xi = F_W / params_.mass; // Eq. 26
  const Eigen::Vector3d abc = xi.normalized(); // Eq. 19

  const double a = abc[0];
  const double b = abc[1];
  const double c = abc[2];
  const double psi = goal.psi;


  const double invsqrt21pc = (1 / std::sqrt(2 * (1 + c)));

  // https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html, order is (w, x, y, z)
  Eigen::Quaterniond q_ref = Eigen::Quaterniond(invsqrt21pc*(1+c), invsqrt21pc*(-b), invsqrt21pc*a, 0.0)
                           * Eigen::Quaterniond(std::cos(psi/2.), 0., 0., std::sin(psi/2.)); // Eq. 14

  // By defn, q_ref has unit norm - but normalize in case numerical issues
  q_ref = q_ref.normalized();


  // TODO: implement the second fibration to cover the whole SO(3).
  // Right now there is a singularity when the drone is upside-down.
  // See Eq. 22, 23 and 24.

  // Log control signals for debugging and inspection
  log_.q = state.q;
  log_.q_ref = q_ref;

  return q_ref;
}

// ----------------------------------------------------------------------------

Eigen::Vector3d OuterLoop::getRates(double dt, const State& state,
                                  const Goal& goal, const Eigen::Vector3d& F_W,
                                  const Eigen::Vector3d& a_fb,
                                  const Eigen::Quaterniond& q_ref)
{
  // For referenced equations, see
  // "Control of Quadrotors Using the Hopf Fibration on SO(3)",
  // https://link.springer.com/chapter/10.1007/978-3-030-28619-4_20

  //
  // Generate feedback jerk by via numerical derivative of feedback accel
  //

  // numerically differentiate accel feedback if possible
  // TODO: could be avoided if we had accel in State (and then apply Leibniz integral rule for the integral term)
  Eigen::Vector3d j_fb;
  if (dt > 0) {
    // take numeric derivative
    j_fb = (a_fb - a_fb_last_) / dt;

    // low-pass file differentiation with time constant tau [sec]
    static constexpr double tau = 0.1;
    const double alpha = dt / (tau + dt);
    j_fb = alpha * j_fb + (1 - alpha) * j_fb_last_;
  } else {
    // simply re-use last value
    j_fb = j_fb_last_;
  }

  // save for next time
  a_fb_last_ = a_fb;
  j_fb_last_ = j_fb;


  //
  // Construct angular rates consistent with trajectory dynamics
  //

  // helper intermediate values
  const Eigen::Vector3d Fdot_W = params_.mass * (goal.j + j_fb);
  const Eigen::Vector3d xi = F_W / params_.mass;  // Eq. 26
  const Eigen::Vector3d abc = xi.normalized();    // Eq. 19
  const Eigen::Vector3d xi_dot = Fdot_W / params_.mass;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const double norm_xi = xi.norm();

  Eigen::Vector3d abcdot =
      ((std::pow(norm_xi, 2) * I - xi * (xi.transpose())) / std::pow(norm_xi, 3)) *
      xi_dot; // Eq. 20, see also https://math.stackexchange.com/questions/2983445/unit-vector-differentiation

  // Note that abc'*abcdot should be approximately 0.0, because we are differentiating a unit vector
  assert(abc.dot(abcdot).isApprox(0.0));

  const double a = abc[0];
  const double b = abc[1];
  const double c = abc[2];

  const double adot = abcdot[0];
  const double bdot = abcdot[1];
  const double cdot = abcdot[2];
  const double psi = goal.psi;
  const double psidot = goal.dpsi;

  Eigen::Vector3d rates; // Eq. 16
  rates.x() = std::sin(psi) * adot - std::cos(psi) * bdot - (a * std::sin(psi) - b * std::cos(psi)) * (cdot / (c + 1));
  rates.y() = std::cos(psi) * adot + std::sin(psi) * bdot - (a * std::cos(psi) + b * std::sin(psi)) * (cdot / (c + 1));
  rates.z() = (b * adot - a * bdot) / (1 + c) + psidot;

  // TODO: implement the second fibration to cover the whole SO(3).
  // Right now there is a singularity when the drone is upside-down.
  // See Eq. 25

  // Log control signals for debugging and inspection
  log_.j_ff = goal.j;
  log_.j_fb = j_fb;
  log_.w = state.w;
  log_.w_ref = rates;

  return rates;
}

}  // namespace outer_loop
}  // namespace acl
