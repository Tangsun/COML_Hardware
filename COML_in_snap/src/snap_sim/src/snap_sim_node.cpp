/**
 * @file snap_sim_node.cpp
 * @brief Entry point for ROS snap stack simulation
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 18 December 2019
 */

#include <iostream>

#include <ros/ros.h>

#include "snap_sim/snap_sim.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "snap_sim");
  ros::NodeHandle nhtopics("");
  ros::NodeHandle nhparams("~");
  acl::snap_sim::SnapSim quad(nhtopics, nhparams);
  ros::spin();
  return 0;
}
