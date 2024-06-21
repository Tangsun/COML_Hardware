/**
 * @file outer_loop_node.cpp
 * @brief Entry point for snap-stack outer-loop controller
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 12 August 2020
 */

#include <iostream>

#include <ros/ros.h>

#include <outer_loop/outer_loop_ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "outer_loop");
  ros::NodeHandle nhtopics("");
  ros::NodeHandle nhparams("~");
  acl::outer_loop::OuterLoopROS node(nhtopics, nhparams);
  ros::spin();
  return 0;
}
