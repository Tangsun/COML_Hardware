/**
 * @file physics_engine.cpp
 * @brief Physics simulation engine containing RigidBody objects
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 9 April 2020
 *  
 *  NOTE: Coordinate frames are ENU/flu.
 */

#include "snap_sim/physics_engine.h"

namespace acl {
namespace snap_sim {

PhysicsEngine::PhysicsEngine()
: sim_time_(0)
{}

// ----------------------------------------------------------------------------

void PhysicsEngine::addRigidBody(const RigidBodyPtr& body)
{
  rigid_bodies_.push_back(body);
}

// ----------------------------------------------------------------------------

void PhysicsEngine::setSimTime(double t)
{
  sim_time_ = t;
}

// ----------------------------------------------------------------------------

void PhysicsEngine::step(double dt)
{
  // simulate each of the rigid bodies in the world
  for (auto&& rb : rigid_bodies_) {
    rb->simulate(dt, sim_time_);
  }
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

} // ns snap_sim
} // ns acl
