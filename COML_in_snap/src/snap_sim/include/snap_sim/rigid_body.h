/**
 * @file rigid_body.h
 * @brief Simulates rigid body dynamics given an input wrench
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 9 April 2020
 *  
 *  NOTE: Coordinate frames are ENU/flu.
 */

#pragma once

#include <memory>
#include <valarray>
#include <vector>

#include <Eigen/Dense>

namespace acl {
namespace snap_sim {

  static const Eigen::Vector3d GRAVITY = -9.80665 * Eigen::Vector3d::UnitZ();

  class RigidBody
  {
  public:
    /**
     * @brief      Kinematic state associated with 3D rigid body
     */
    struct State {
      double t; ///< timestamp of these kinematics [s]
      Eigen::Vector3d p; ///< p_WB (position of body w.r.t world)
      Eigen::Vector3d v; ///< v_WB (velocity of body w.r.t world)
      Eigen::Quaterniond q; ///< q_WB (orientation of body w.r.t world)
      Eigen::Vector3d w; ///< w_WB_B (angular vel of body w.r.t world expressed in body)
    };

    using StateVec = std::valarray<double>; ///< for numerical rk4 operations
    using Wrench = Eigen::Matrix<double, 6, 1>; ///< forces and then moments
    using Inertia = Eigen::Matrix3d;

  public:
    RigidBody(const State& state0, double mass, const Inertia& J);
    ~RigidBody() = default;

    /**
     * @brief      Integrate forward the rigid body
     *
     * @param[in]  dt    timestep to step by
     * @param[in]  t     the current simulation time
     */
    void simulate(double dt, double t);

    /**
     * @brief      State getter.
     *
     * @return     Constant reference to the kinematic state
     */
    const State& getState() const { return state_; }
    
  protected:
    State state_; ///< current kinematic state

    /// \brief Inertial properties
    double mass_; ///< total mass [kg]
    Inertia J_; ///< inertia matrix

    bool grounded_; ///< have we produced a wrench to overcome our weight?
    Wrench current_wrench_; ///< current wrench exerted on body, in body frame.

    /**
     * @brief      Is this rigid body allowed to generate wrench when grounded?
     *
     * @return     True if able to move on ground, False otherwise.
     */
    virtual bool canMoveOnGround() const { return true; }

    /**
     * @brief      Calculate the wrench from external forces acting on the body.
     *
     * @param[in]  dt     the simulation step time
     * 
     * @return     The external wrench exterted on the body, in the body frame.
     */
    virtual Wrench getWrench(double dt) = 0;

    /**
     * @brief      Rigid body dynamics, ydot = f(t,y)
     *
     * @param[in]  t        current time
     * @param[in]  y        current state
     * @param[in]  wrench   input wrench that drives the rigid body
     *
     * @return     derviatives of the state, ydot
     */
    virtual StateVec f(double t, const StateVec& y, const Wrench& wrench);

    /**
     * @brief      Converts state representations. From StateVec to State.
     *
     * @param[in]  y     the StateVec to convert from
     *
     * @return     Represented as a state object
     */
    static State toState(const StateVec& y);

    /**
     * @brief      Converts state representations. From State to StateVec.
     *
     * @param[in]  y     the State to convert from
     *
     * @return     Represented as a state vector
     */
    static StateVec fromState(const State& state);

  private:
    using DynamicsFcn = std::function<StateVec(double, const StateVec&)>;

    double floorPlane_; ///< z position of floor (infinitely extending plane)

    /**
     * @brief      Computes the total wrench acting on the rigid body as input
     *             into the system dynamics, f(t, y).
     *             
     *             This includes normal force from the floor and does not
     *             include gravity (which is included in the model, f).
     *
     * @param[in]  dt    Time step
     *
     * @return     The total wrench, expressed in the body frame
     */
    Wrench getTotalWrench(double dt);

    /**
     * @brief      Checks the kinematic state to see if rigid body is in
     *             collision with the floor, which is an infinite plane.
     */
    void checkFloorCollision();

    /**
     * @brief      Integrate coupled first-order ODE via fourth-order Runge-Kutta
     *
     * @param[in]  dt    timestep
     * @param[in]  t     current time
     * @param[in]  y     current state
     * @param[in]  f     system dynamics function that returns derivatives
     *
     * @return     Integrated state vector
     */
    static StateVec rk4(double dt, double t, const StateVec& y, DynamicsFcn f);
  };

  using RigidBodyPtr = std::shared_ptr<RigidBody>;

} // ns snap_sim
} // ns acl
