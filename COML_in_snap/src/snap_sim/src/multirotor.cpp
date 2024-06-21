/**
 * @file multirotor.cpp
 * @brief RigidBody specialization with multirotor geometry, actuators, sensors
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 9 April 2020
 *
 * Inpsired by 
 *  - ROSflight (https://github.com/rosflight/rosflight/blob/master/rosflight_sim/src/multirotor_forces_and_moments.cpp)
 *  - A. Torgesen (https://github.com/goromal/air_dynamics/blob/master/src/air_dynamics.cpp)
 *  
 *  NOTE: Coordinate frames are ENU/flu.
 */

#include "snap_sim/multirotor.h"

namespace acl {
namespace snap_sim {

Multirotor::Multirotor(const State& state0, const Params& params)
: state0_(state0), params_(params), rnd_((std::random_device())()),
  RigidBody(state0, params.mass, params.J)
{
  //
  // Use multirotor geometry to build normalized wrench maps
  //

  // initialize maps based on number of actuators
  wrench_from_motor_thrust_map_ = WrenchMap::Zero(6, params_.motors.size());
  wrench_from_motor_torque_map_ = WrenchMap::Zero(6, params_.motors.size());

  for (size_t i=0; i<params_.motors.size(); ++i) {
    auto& motor = params_.motors[i];

    // make sure that thrust vector is unit length
    motor.direction.normalize();

    // calculate position of motor w.r.t com of vehicle
    const auto pos_wrt_com = motor.position - params_.center_of_mass;

    // calculate moments generated from motor thrust vectors
    const auto& body_moment_from_thrust = pos_wrt_com.cross(motor.direction);
    const auto& body_moment_from_drag_torque = -1 * motor.spin * motor.direction;

    //
    // Build normalized wrench maps
    //

    // maps motor thrusts to body wrench
    wrench_from_motor_thrust_map_.block<3,1>(0,i) = motor.direction;
    wrench_from_motor_thrust_map_.block<3,1>(3,i) = body_moment_from_thrust;

    // maps motor drag torques to body wrench
    wrench_from_motor_torque_map_.block<3,1>(0,i) = Eigen::Vector3d::Zero();
    wrench_from_motor_torque_map_.block<3,1>(3,i) = body_moment_from_drag_torque;

    dryden_wind_.initialize(params_.wind_nominal.x(), params_.wind_nominal.y(), params_.wind_nominal.z(),
                            params_.wind_gust_bound.x(), params_.wind_gust_bound.y(), params_.wind_gust_bound.z());
  }

}

// ----------------------------------------------------------------------------

void Multirotor::setMotorCommands(const MotorCmds& u)
{
  for (size_t i=0; i<NUM_PWM; ++i) u_[i] = u[i];
}

// ----------------------------------------------------------------------------

double Multirotor::readIMU(Eigen::Vector3d& acc, Eigen::Vector3d& gyr)
{
  // NOTE: gravity is not in the wrench because it is in the EoM.
  acc = (current_wrench_.head<3>() / params_.mass) /*- GRAVITY*/;
  gyr = state_.w;

  // add IMU noise, which is mostly due to the motors spinning
  if (motorsSpinning()) {
    acc += getNoiseWithWalkingBias(bias_accel_,
                              params_.accel_stdev, params_.accel_walk_stdev);
    gyr += getNoiseWithWalkingBias(bias_gyro_,
                              params_.gyro_stdev, params_.gyro_walk_stdev);
  }

  return state_.t;
}

// ----------------------------------------------------------------------------
// Protected Methods
// ----------------------------------------------------------------------------

RigidBody::Wrench Multirotor::getWrench(double dt)
{
  // Get wind in world frame
  Eigen::Vector3d wind = received_wind_; //getWind(dt);
  // std::cout << received_wind_;
  // Use wind triangle to calculate airspeed vector
  Eigen::Vector3d Va = state_.q.inverse() * (state_.v - wind);  // in body frame

  // Calculate wrenches acting on body because of motors
  const Wrench motors = getWrenchDueToMotors(dt);
  const Wrench body_drag = getWrenchDueToDrag(Va, state_.w);
  const Wrench ground_effect = getWrenchDueToGroundEffect(state_.p.z());
  const Wrench blade_flapping = getWrenchDueToBladeFlapping();

  // Combine all wrenches except acting on body because of motors
  const Wrench wrench = motors + body_drag + ground_effect + blade_flapping;

  return wrench;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

double Multirotor::evalPoly(const std::vector<double>& coeffs, double x)
{
  // assumption: coeffs is ordered as [a_n ... a_0] s.t y = a_n*x^n + ... + a_0
  double y = 0.0;
  for (size_t i=0; i<coeffs.size(); ++i) {
    y += coeffs[i] * std::pow(x, coeffs.size() - 1 - i);
  }
  return y;
}

// ----------------------------------------------------------------------------

bool Multirotor::motorsSpinning()
{
  static constexpr double MOTOR_IDLE_SPIN = 0.05;
  for (size_t i=0; i<params_.motors.size(); ++i) {
    if (u_[i] > MOTOR_IDLE_SPIN) {
      return true;
    }
  }

  return false;
}

// ----------------------------------------------------------------------------

Eigen::Vector3d Multirotor::getNoiseWithWalkingBias(Eigen::Vector3d& bias,
                                        double stdev, double bias_walk_stdev)
{

  Eigen::Vector3d zero_mean_noise;
  zero_mean_noise.x() = stdev*normal_dist_(rnd_);
  zero_mean_noise.y() = stdev*normal_dist_(rnd_);
  zero_mean_noise.z() = stdev*normal_dist_(rnd_);

  Eigen::Vector3d bias_walk;
  bias.x() += bias_walk_stdev*normal_dist_(rnd_);
  bias.y() += bias_walk_stdev*normal_dist_(rnd_);
  bias.z() += bias_walk_stdev*normal_dist_(rnd_);

  return bias + zero_mean_noise;
}

// ----------------------------------------------------------------------------

RigidBody::Wrench Multirotor::getWrenchDueToMotors(double dt)
{
  static Eigen::Matrix<double, Eigen::Dynamic, 1> des_motor_thrusts, des_motor_torques;
  static Eigen::Matrix<double, Eigen::Dynamic, 1> cur_motor_thrusts, cur_motor_torques;

  // initialize data structures based on num actuators
  if (des_motor_thrusts.size() == 0) {
    const size_t m = params_.motors.size();
    des_motor_thrusts = Eigen::MatrixXd::Zero(m, 1);
    des_motor_torques = Eigen::MatrixXd::Zero(m, 1);
    cur_motor_thrusts = Eigen::MatrixXd::Zero(m, 1);
    cur_motor_torques = Eigen::MatrixXd::Zero(m, 1);
  }

  // build two m-length vectors: motor trusts and motor torques
  for (size_t i=0; i<params_.motors.size(); ++i) {
    // for convenience
    const auto& motor = params_.motors[i];
    const double u = u_[i];

    // map requested pwm into motor thrusts and torques
    des_motor_thrusts(i) = evalPoly(motor.rotor.polyF, u);
    des_motor_torques(i) = evalPoly(motor.rotor.polyT, u);

    // filter through first-order spin up / spin down dynamics
    const double tau = (des_motor_thrusts(i) > cur_motor_thrusts(i)) ? motor.rotor.tauUp : motor.rotor.tauDn;
    const double alpha = dt / (tau + dt);
    cur_motor_thrusts(i) = alpha*des_motor_thrusts(i) + (1-alpha)*cur_motor_thrusts(i);
    cur_motor_torques(i) = alpha*des_motor_torques(i) + (1-alpha)*cur_motor_torques(i);

    // clamp the output to respect actuator limits
    cur_motor_thrusts(i) = acl::clamp(cur_motor_thrusts(i), motor.rotor.minF, motor.rotor.maxF);
    cur_motor_torques(i) = acl::clamp(cur_motor_torques(i), motor.rotor.minT, motor.rotor.maxT);
  }

  // map motor thrusts and torques into body wrenches
  const Wrench wrench_from_motor_thrusts = wrench_from_motor_thrust_map_ * cur_motor_thrusts;
  const Wrench wrench_from_motor_torques = wrench_from_motor_torque_map_ * cur_motor_torques;

  return wrench_from_motor_thrusts + wrench_from_motor_torques;
}

// ----------------------------------------------------------------------------

RigidBody::Wrench Multirotor::getWrenchDueToDrag(const Eigen::Vector3d& Va,
                                                  const Eigen::Vector3d& omega)
{
  Wrench drag;
  drag.head<3>() = -(params_.muVLin*Va + params_.muVQuad*Va.norm()*Va);
  drag.tail<3>() = -params_.muOmega*omega.norm()*omega;

  return drag;
}

// ----------------------------------------------------------------------------

RigidBody::Wrench Multirotor::getWrenchDueToGroundEffect(double z)
{
  // plot: https://github.com/byu-magicc/fcu_sim/pull/38
  const double lift = evalPoly(params_.polyGroundEffect, z);

  Wrench ground_effect = Wrench::Zero();
  ground_effect.head<3>().z() = acl::clamp(lift, 0.0, 1000.0); // ensure >0

  return ground_effect;
}

// ----------------------------------------------------------------------------

RigidBody::Wrench Multirotor::getWrenchDueToBladeFlapping()
{
  // see A. Torgesen's notes https://wiki-notes.000webhostapp.com/doku.php?id=public:waslanderbladeflapping

  return Wrench::Zero();
}

// ----------------------------------------------------------------------------

Eigen::Vector3d Multirotor::getWind(double dt)
{
  return dryden_wind_.getWind(dt);
}

} // ns snap_sim
} // ns acl
