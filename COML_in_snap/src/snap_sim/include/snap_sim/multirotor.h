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

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "snap_sim/rigid_body.h"
#include "snap_sim/utils.h"
#include "wind-dynamics/dryden_model.h"

namespace acl {
namespace snap_sim {

  class Multirotor : public RigidBody
  {
  public:

    static constexpr size_t NUM_PWM = 8;
    using MotorCmds = std::array<double, NUM_PWM>;

    /**
     * @brief      Physical properties of motor+prop+esc propulsion unit
     */
    struct Rotor {
      std::vector<double> polyF; ///< poly coeffs: map normalized pwm to force
      std::vector<double> polyT; ///< poly coeffs: map pwm usec to drag torque
      double minF, maxF; ///< minimum / maximum achievable force [N]
      double minT, maxT; ///< minimum / maximum achievable torque [Nm]
      double tauUp; ///< time constant for first-order motor spin up response
      double tauDn; ///< time constant for first-order motor spin down response
    };

    /**
     * @brief      Geometry of propulsion unit. Expressed in body frame.
     */
    struct Motor {
      Rotor rotor; ///< physical force and torque properties
      Eigen::Vector3d position; ///< motor thrust point w.r.t body origin
      Eigen::Vector3d direction; ///< unit vector defining thrust vector
      int spin; ///< direction of prop spin: CW(-1) or CCW(1)
    };

    /**
     * @brief      Parameters to define multirotor configuration and wind
     */
    struct Params {
      double mass; ///< total multirotor mass [kg]
      RigidBody::Inertia J; ///< total inertia of multirotor
      std::vector<Motor> motors;
      double muVLin, muVQuad, muOmega; ///< drag coefficients
      std::vector<double> polyGroundEffect; ///< poly coeffs to map altitude
                                            ///< to ground effect force
      Eigen::Vector3d center_of_mass; ///< w.r.t body origin (geometric center)
      /// \brief Sensors
      double gyro_stdev;
      double gyro_walk_stdev;
      double accel_stdev;
      double accel_walk_stdev;
      /// \brief Wind disturbance
      Eigen::Vector3d wind_nominal; ///< nominal wind velocity [m/s]
      Eigen::Vector3d wind_gust_bound; ///< loose bound on the additional gust velocity [m/s]
    };

  public:
    Multirotor(const State& state0, const Params& params);
    ~Multirotor() = default;
    
    /**
     * @brief      Simulates writing the motor commands to the motors.
     *
     * @param[in]  u     The motor commands in normalized pwm, \in [0 1]
     */
    void setMotorCommands(const MotorCmds& u);

    /**
     * @brief      Reads the simulated IMU. Note that the IMU is oriented
     *             exactly the same as the body (i.e., identity rotation).
     *
     * @param      acc   The specific force measured by an accelerometer
     * @param      gyr   The angular velocity measured by a gyroscope
     * 
     * @return     Timestamp in seconds of IMU reading
     */
    double readIMU(Eigen::Vector3d& acc, Eigen::Vector3d& gyr);

  
    void setWind(const Eigen::Vector3d& wind){
      // check wind value within range?
      received_wind_ = wind; 
    };

  protected:

    /**
     * @brief      Is this rigid body allowed to generate wrench when grounded?
     *
     * @return     True if able to move on ground, False otherwise.
     */
    bool canMoveOnGround() const override { return false; }

    /**
     * @brief      Calculate the wrench acting on the body, in the body frame.
     *
     * @param[in]  dt    the simulation step time
     * 
     * @return     The external wrench exterted on the body, in the body frame.
     */
    RigidBody::Wrench getWrench(double dt) override;

  private:
    State state0_; ///< initial kinematic state
    Params params_; ///< inertial, geometric, and actuator properties

    Eigen::Vector3d received_wind_ = Eigen::Vector3d::Zero(); // wind speed received from the subcriber. 

    MotorCmds u_; ///< normalized pwm commands, in [0, 1]

    /// \brief Maps from motor wrench to body wrench
    using WrenchMap = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    WrenchMap wrench_from_motor_thrust_map_;
    WrenchMap wrench_from_motor_torque_map_;

    std::mt19937 rnd_; ///< random number generator
    std::normal_distribution<double> normal_dist_; ///< normal distribution sampler

    Eigen::Vector3d bias_gyro_, bias_accel_; ///< slowly-varying IMU biases

    // Wind simulation
    dryden_model::DrydenWind dryden_wind_;  ///< wind velocity generator

    /**
     * @brief      Given a set of coefficients, evaluates a polynomial
     *
     * @param[in]  coeffs  The polynomial coefficients [a_n ... a_0]
     * @param[in]  x       The independent variable
     *
     * @return     y = a_n*x^n + ... + a_0
     */
    double evalPoly(const std::vector<double>& coeffs, double x);

    bool motorsSpinning();

    Eigen::Vector3d getNoiseWithWalkingBias(Eigen::Vector3d& bias,
                                double stdev, double bias_walk_stdev);

    Eigen::Vector3d getWind(double dt);

    RigidBody::Wrench getWrenchDueToMotors(double dt);

    RigidBody::Wrench getWrenchDueToDrag(const Eigen::Vector3d& Va,
                              const Eigen::Vector3d& omega);

    RigidBody::Wrench getWrenchDueToGroundEffect(double z);

    RigidBody::Wrench getWrenchDueToBladeFlapping();
  };

  using MultirotorPtr = std::shared_ptr<Multirotor>;

} // ns snap_sim
} // ns acl
