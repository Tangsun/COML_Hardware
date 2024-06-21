/****************************************************************************
 *   Copyright (c) 2017 Brett T. Lopez. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name snap nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

#include <string>
#include <algorithm>
#include <cctype>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/convert.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/Float32.h>

// Custom messages
#include <snapstack_msgs/AttitudeCommand.h>
#include <snapstack_msgs/IMU.h>
#include <snapstack_msgs/State.h>
#include <snapstack_msgs/SMCData.h>
#include <snapstack_msgs/Motors.h>

#include "SnapdragonObserverManager.hpp"

#ifdef SNAP_APM
#include <snap_apm/snap_apm.h>
#endif


/**
 * Wrapper Ros Node to support observer from Snapdragon flight platform.
 */
namespace Snapdragon {
  namespace RosNode {
    class SND;
  }
}

/**
 * Ros SND Node that uses Snapdragon platform to get SND pose data.
 */
class Snapdragon::RosNode::SND
{
public:
  /**
   * Constructor.
   * @param nh
   *   Ros Node handle to intialize the node.
   */
  SND( ros::NodeHandle nh );

  /**
   * Start the SND node processing.
   * @return int32_t
   *  0 = success
   * otherwise = false;
   **/
  int32_t Start();

  /**
   * Stops the SND processing thread.
   * @return int32_t
   *  0 = success;
   * otherwise = false;
   **/
  int32_t Stop();

  /**
   * Get the Params checking that they exist (and don't resort to default values).
   */
  template <typename T>
  bool safeGetParam(ros::NodeHandle& nh, std::string const& param_name, T& param_value)
  {
    if (!nh.getParam(param_name, param_value))
    {
      ROS_ERROR("Failed to find parameter: %s", nh.resolveName(param_name, true).c_str());
      exit(1);
    }
    return true;
  }

  /**
   * Destructor for the node.
   */
  ~SND();

  float lin_acc[3], ang_vel[3];
  uint32_t sequence_number;
  uint64_t current_timestamp_ns;

private: 
  // class methods
  bool   armCB(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
  bool   isarmedCB(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
  void   PublishIMUData(const Snapdragon::ObserverManager::Data& ukf_data );
  void   PublishStateData(const Snapdragon::ObserverManager::State& ukf_state );
  void   PublishOdometryData(const Snapdragon::ObserverManager::State& ukf_state );
  void   PublishMotorCommands (const Snapdragon::ControllerManager::motorThrottles& throttles );
  void   PublishSMCData(const Snapdragon::ControllerManager::controlData& data );
  void   BroadcastTF (const Snapdragon::ObserverManager::State& ukf_state );
  void   poseCB(const geometry_msgs::PoseStamped& msg);
  void   goalCB(const snapstack_msgs::AttitudeCommand& msg);
  void   pubCB (const ros::TimerEvent& e);
  void   getParams(Snapdragon::ObserverManager::InitParams& Params, Snapdragon::ControllerManager::InitParams& SmcParams);
  void   vec2ROS(geometry_msgs::Vector3& vros, const Vector& v);
  void   vec2ROS(geometry_msgs::Vector3& vros, const tf2::Vector3& v);
  void   vec2ROS( geometry_msgs::Point& vros, const Vector& v);
  void   quat2ROS(geometry_msgs::Quaternion& qros, const Quaternion& q);
  void   ROS2vec(Vector& v, const geometry_msgs::Point& vros);
  void   ROS2vec(Vector& v, const geometry_msgs::Vector3& vros);
  void   ROS2quat(Quaternion& q, const geometry_msgs::Quaternion& qros);

  // data members
  std::string vehname_; ///< vehicle name
  bool broadcast_tf_;
  Snapdragon::ObserverManager observer_man_;
  ros::NodeHandle   nh_;
  ros::Publisher    pub_imu_;
  ros::Publisher    pub_state_;
  ros::Publisher    pub_odom_;
  ros::Publisher    pub_motor_;
  ros::Publisher    pub_smc_;
  ros::Subscriber   sub_pose_;
  ros::Subscriber   sub_attCmd_;
  ros::Timer        pubDataTimer_ ;
  ros::ServiceServer srv_arm_, srv_isarmed_;
  tf2_ros::TransformBroadcaster br_;

  // Snap APM
#ifdef SNAP_APM
  // APM publisher function, objects, and timer
  void pubApmCB(const ros::TimerEvent& e);
  ros::Publisher pub_vol_, pub_curr_;
  ros::Timer     pub_apm_timer_;
  // SnapAPM object
  acl::SnapAPM apm_;
#endif

};
