/**
 * @file rigid_body.cpp
 * @brief Simulates rigid body dynamics given an input wrench
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 9 April 2020
 *  
 *  NOTE: Coordinate frames are ENU/flu.
 */

#include "snap_sim/rigid_body.h"
#include <iostream>

namespace acl {
namespace snap_sim {

using StateVec = RigidBody::StateVec;
using State = RigidBody::State;
using Wrench = RigidBody::Wrench;

RigidBody::RigidBody(const State& state0, double mass, const Inertia& J)
: state_(state0), mass_(mass), J_(J), grounded_(false),
  current_wrench_(Wrench::Zero()), floorPlane_(0.0)
{
  //std::cout << "rigid body state_0: " << state0.q.w() << std::endl;
  //std::cout << "rigid body state: " << state_.q.w() << std::endl;
  checkFloorCollision();
}

// ----------------------------------------------------------------------------

void RigidBody::simulate(double dt, double t)
{
  //std::cout << "state_ bfr rk: " << state_.q.w() << std::endl;
  const auto& y0 = fromState(state_);

  // get total wrench acting on body (expressed in body frame)
  const auto& wrench = getTotalWrench(dt);

  // TODO: this rk4 assumes everything lives in a Euclidean vector space. The
  // quaternion portion, of course, does not. Though not principled, this is a
  // decent thing to do provided the timestep is small. As a result, we must
  // ensure that the quaternion is unit length after the integration step.
  const auto y1 = rk4(dt, t, y0, std::bind(&RigidBody::f, this,
                            std::placeholders::_1, std::placeholders::_2,
                            wrench));

  state_ = toState(y1);
  //std::cout << "state_ aftr rk: " << state_.q.w() << std::endl;
  state_.t = t;

  checkFloorCollision();

  // normalize quaternion
  state_.q.normalize();
}

// ----------------------------------------------------------------------------
// Protected Members
// ----------------------------------------------------------------------------

StateVec RigidBody::f(double t, const StateVec& y, const Wrench& wrench)
{
  const auto state = toState(y);

  // pure quaternion from angular velocity
  Eigen::Quaterniond omegabar;
  omegabar.w() = 0;
  omegabar.vec() = state.w;

  Eigen::Vector3d pdot = state.v;
  Eigen::Vector3d vdot = state.q * ( 1.0/mass_ * wrench.head<3>() ) + GRAVITY;
  Eigen::Quaterniond qdot = state.q * omegabar; qdot.coeffs() *= 0.5;
  Eigen::Vector3d wdot = J_.inverse()*((J_ * state.w).cross(state.w) + wrench.tail<3>());

  return fromState(State{t, pdot, vdot, qdot, wdot});
}

// ----------------------------------------------------------------------------

State RigidBody::toState(const StateVec& y)
{
  State state;
  state.p.x() = y[0];
  state.p.y() = y[1];
  state.p.z() = y[2];
  state.v.x() = y[3];
  state.v.y() = y[4];
  state.v.z() = y[5];
  state.q.w() = y[6];
  state.q.x() = y[7];
  state.q.y() = y[8];
  state.q.z() = y[9];
  state.w.x() = y[10];
  state.w.y() = y[11];
  state.w.z() = y[12];
  return state;
}

// ----------------------------------------------------------------------------

StateVec RigidBody::fromState(const State& state)
{
  StateVec y(13);
  y[0] = state.p.x();
  y[1] = state.p.y();
  y[2] = state.p.z();
  y[3] = state.v.x();
  y[4] = state.v.y();
  y[5] = state.v.z();
  y[6] = state.q.w();
  y[7] = state.q.x();
  y[8] = state.q.y();
  y[9] = state.q.z();
  y[10] = state.w.x();
  y[11] = state.w.y();
  y[12] = state.w.z();
  return y;
}

// ----------------------------------------------------------------------------
// Private Members
// ----------------------------------------------------------------------------

Wrench RigidBody::getTotalWrench(double dt)
{
  // calculate normal force from floor (assumes flat floor)
  static const Eigen::Vector3d Fnormal = - mass_ * GRAVITY;

  // get wrench (in body frame) due to actuation and other modelled forces
  const Wrench body = getWrench(dt);

  // rotate into world frame (will likely be the same)
  const Eigen::Vector3d Fworld = state_.q * body.head<3>();

  //
  // Check if rigid body has overcome its own weight
  //

  if (grounded_) {
    if (Fworld.z() > Fnormal.z()) {
      grounded_ = false;
    }
  }

  //
  // Determine what the force due to floor interaction should be
  //

  Wrench total = body;

  // if ground locked, zero appropriate wrench elements and add normal force
  if (grounded_) {
    if (canMoveOnGround()) {
      // null any force in world z
      Eigen::Vector3d tmp = Fworld;
      tmp.z() = Fnormal.z();
      total.head<3>() = state_.q.inverse() * tmp;

      // null any moment in body x / y.
      total.tail<3>().x() = 0;
      total.tail<3>().y() = 0;
    } else {
      total = Wrench::Zero();
      total.head<3>() = state_.q.inverse() * Fnormal;
    }
  }

  // set the last total wrench used (e.g., for accel calculation)
  current_wrench_ = total;

  // NOTE: gravity is not in the wrench because it is in the EoM.
  return total;
}

// ----------------------------------------------------------------------------

void RigidBody::checkFloorCollision()
{

  if (!grounded_) {
    if (state_.p.z() <= floorPlane_) {
      state_.p.z() = floorPlane_;
      state_.v.x() /= 1000.0;
      state_.v.y() /= 1000.0;
      state_.v.z() = 0;

      // always land level (no roll/pitch)
      // const auto& q = state_.q;
      // const auto yaw = std::atan2(-2*q.x()*q.y() + 2*q.w()*q.z(), +q.w()*q.w() +q.x()*q.x() -q.y()*q.y() -q.z()*q.z());
      // state_.q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

      // always land with identity orientation
      //state_.q = Eigen::Quaterniond::Identity();

      state_.w.x() = 0;
      state_.w.y() = 0;
      state_.w.z() = 0;

      grounded_ = true;
    }
  }

}

// ----------------------------------------------------------------------------

StateVec RigidBody::rk4(double dt, double t, const StateVec& y, DynamicsFcn f)
{
  StateVec k1 = dt * f(t, y);
  StateVec k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1);
  StateVec k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2);
  StateVec k4 = dt * f(t + dt, y + k3);
  return y + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}


} // ns snap_sim
} // ns acl
