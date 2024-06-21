/**
 * @file physics_engine.h
 * @brief Physics simulation engine containing RigidBody objects
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 9 April 2020
 *  
 *  NOTE: Coordinate frames are ENU/flu.
 */

#pragma once

#include <vector>

#include "snap_sim/rigid_body.h"

namespace acl {
namespace snap_sim {

  class PhysicsEngine
  {
  public:
    PhysicsEngine();
    ~PhysicsEngine() = default;

    /**
     * @brief      Add a rigid body to the physics world for simulation
     *
     * @param[in]  body  The rigid body to add
     */
    void addRigidBody(const RigidBodyPtr& body);

    /**
     * @brief      Overwrites the current sim time
     *
     * @param[in]  t     the simulation time to use
     */
    void setSimTime(double t);

    /**
     * @brief      Step forward the physics world
     *
     * @param[in]  dt    Timestep to simulate forward with
     */
    void step(double dt);
    
  private:
    std::vector<RigidBodyPtr> rigid_bodies_; ///< physics bodies in my world

    double sim_time_; ///< current simulation time
  };

} // ns snap_sim
} // ns acl
