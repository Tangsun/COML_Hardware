/***************************************************************************r
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

#include "SnapdragonRosNode.hpp"
#include "SnapdragonDebugPrint.h"
#include "SnapdragonUtils.hpp"

Snapdragon::RosNode::SND::SND( ros::NodeHandle nh ) : nh_(nh)
{
  // Get vehicle name
  vehname_ = ros::this_node::getNamespace();
  vehname_.erase(0, vehname_.find_first_not_of('/')); // remove leading slashes

  pub_imu_   = nh_.advertise<snapstack_msgs::IMU>("imu",1);
  pub_state_ = nh_.advertise<snapstack_msgs::State>("state",1);
  pub_odom_  = nh_.advertise<nav_msgs::Odometry>("odometry",1);
  pub_motor_ = nh_.advertise<snapstack_msgs::Motors>("motors",1);
  pub_smc_   = nh_.advertise<snapstack_msgs::SMCData>("smc",1);

  sub_pose_   = nh_.subscribe("pose", 1, &Snapdragon::RosNode::SND::poseCB, this);
  sub_attCmd_ = nh_.subscribe("attcmd", 1, &Snapdragon::RosNode::SND::goalCB, this);

  srv_arm_ = nh_.advertiseService("arm", &Snapdragon::RosNode::SND::armCB, this);
  srv_isarmed_ = nh_.advertiseService("is_armed", &Snapdragon::RosNode::SND::isarmedCB, this);

  pubDataTimer_ = nh_.createTimer(ros::Duration(0.005), &Snapdragon::RosNode::SND::pubCB, this);

#ifdef SNAP_APM
  ROS_INFO("Initializing Advanced Power Module (APM)...");
  apm_.init();
  if(not apm_.hw_error()){
    ROS_INFO("Advanced Power Module (APM) initialized");
    pub_vol_  = nh_.advertise<std_msgs::Float32>("battery/voltage",1);
    pub_curr_ = nh_.advertise<std_msgs::Float32>("battery/current",1);
    pub_apm_timer_ = nh_.createTimer(ros::Duration(0.1), &Snapdragon::RosNode::SND::pubApmCB, this);
  }
  else
    ROS_ERROR("The Advanced Power Module could not be initialized");
#endif

  // sleep here so tf buffer can get populated
  ros::Duration(1).sleep(); // sleep for 1 second
}

Snapdragon::RosNode::SND::~SND()
{
  Stop();
}

int32_t Snapdragon::RosNode::SND::Start() {
  Snapdragon::ObserverManager::InitParams Params;
  Snapdragon::ControllerManager::InitParams SmcParams;

  getParams(Params, SmcParams);

  if( observer_man_.Initialize(vehname_, Params, SmcParams) != 0  ) {
    ROS_WARN_STREAM( "Snapdragon::RosNode: Error initializing the SND Manager" );
    return -1;
  }

// start the snap processing.
  if( observer_man_.Start() != 0 ) {
    ROS_WARN_STREAM( "Snapdragon::RosNode: Error initializing the SND Manager"  );
    return -1;
  }

  return 0;
}

int32_t Snapdragon::RosNode::SND::Stop(){
  // TODO: make sure other process are shutdown properly
#ifdef SNAP_APM
  if (!apm_.hw_error()) apm_.close();
#endif
  return 0;
}

bool Snapdragon::RosNode::SND::armCB(std_srvs::SetBool::Request &req,
                                     std_srvs::SetBool::Response &res)
{
  bool arm_requested = req.data;
  if (arm_requested && observer_man_.isCalibrated()) {
    bool success = observer_man_.escManager()->arm();
    if (!success) res.message = "Failed to arm";
    else res.message = "ARMED";
  } else {
    bool success = observer_man_.escManager()->disarm();
    if (!success) res.message = "Failed to disarm";
    else res.message = "DISARMED";
  }

  res.success = observer_man_.escManager()->isArmed();
  return true;
}

bool Snapdragon::RosNode::SND::isarmedCB(std_srvs::Trigger::Request &req,
                                         std_srvs::Trigger::Response &res)
{
  res.success = observer_man_.escManager()->isArmed();
  return true;
}

void Snapdragon::RosNode::SND::pubCB (const ros::TimerEvent& e){
  // pub stuff
  if (observer_man_.calibrated_){
    PublishIMUData(observer_man_.imu_data_); 
    PublishStateData(observer_man_.state_);
    PublishOdometryData(observer_man_.state_);
    PublishMotorCommands(observer_man_.smc_motors_);
    PublishSMCData(observer_man_.smc_data_);
    if (broadcast_tf_) BroadcastTF(observer_man_.state_);
  }
}

void Snapdragon::RosNode::SND::PublishIMUData(const Snapdragon::ObserverManager::Data& imu_data  ) {
  snapstack_msgs::IMU imu_msg;
  ros::Time frame_time;
  frame_time.sec = (int32_t)(imu_data.current_timestamp_ns/1000000000UL);
  frame_time.nsec = (int32_t)(imu_data.current_timestamp_ns % 1000000000UL);
  imu_msg.header.frame_id = "body";
  imu_msg.header.stamp = ros::Time::now();
  imu_msg.imu_stamp = frame_time;
  imu_msg.header.seq = imu_data.sequence_number;
  imu_msg.accel.x = imu_data.lin_accel[0];
  imu_msg.accel.y = imu_data.lin_accel[1];
  imu_msg.accel.z = imu_data.lin_accel[2];
  imu_msg.gyro.x = imu_data.ang_vel[0];
  imu_msg.gyro.y = imu_data.ang_vel[1];
  imu_msg.gyro.z = imu_data.ang_vel[2];

  pub_imu_.publish(imu_msg);
}

void Snapdragon::RosNode::SND::PublishStateData(const Snapdragon::ObserverManager::State& state ){
  snapstack_msgs::State state_msg;
  ros::Time frame_time;
  frame_time.sec = (int32_t)(state.current_timestamp_ns/1000000000UL);
  frame_time.nsec = (int32_t)(state.current_timestamp_ns % 1000000000UL);
  state_msg.header.frame_id = "world";
  state_msg.header.stamp = ros::Time::now();
  state_msg.header.seq = state.sequence_number;
  state_msg.state_stamp = frame_time;
  vec2ROS(state_msg.pos,state.pos);
  vec2ROS(state_msg.vel,state.vel);
  quat2ROS(state_msg.quat,state.q);
  vec2ROS(state_msg.w,state.w);
  vec2ROS(state_msg.abias,state.accel_bias);
  vec2ROS(state_msg.gbias,state.gyro_bias);
  pub_state_.publish(state_msg);
}

void Snapdragon::RosNode::SND::PublishOdometryData(const Snapdragon::ObserverManager::State& state ){
    nav_msgs::Odometry odom_msg;
    odom_msg.header.frame_id = "world";  // pose coordinate frame
    odom_msg.header.stamp = ros::Time::now();  // snap time instead of IMU time
    odom_msg.header.seq = state.sequence_number;
    odom_msg.child_frame_id = vehname_;  // twist coordinate frame: body
    vec2ROS(odom_msg.pose.pose.position,state.pos);
    quat2ROS(odom_msg.pose.pose.orientation,state.q);

    // transform linear velocity in world frame to body frame
    tf2::Quaternion quat = tf2::Quaternion(state.q.x, state.q.y, state.q.z, state.q.w);
    tf2::Transform T_BW = tf2::Transform(quat).inverse();

    tf2::Vector3 lin_vel = tf2::Vector3(state.vel.x, state.vel.y, state.vel.z);
    tf2::Vector3 lin_vel_body = T_BW*lin_vel;
    vec2ROS(odom_msg.twist.twist.linear, lin_vel_body);

    // the angular vel is already in body frame
    tf2::Vector3 ang_vel_body = tf2::Vector3(state.w.x, state.w.y, state.w.z);
    vec2ROS(odom_msg.twist.twist.angular, ang_vel_body);

    pub_odom_.publish(odom_msg);
}

void Snapdragon::RosNode::SND::PublishMotorCommands(const Snapdragon::ControllerManager::motorThrottles& throttles ){
  static int count = 0;
  snapstack_msgs::Motors motor_msg;
  motor_msg.header.stamp = ros::Time::now();
  motor_msg.header.seq = count;
  motor_msg.m1 = throttles.throttle[0];
  motor_msg.m2 = throttles.throttle[1];
  motor_msg.m3 = throttles.throttle[2];
  motor_msg.m4 = throttles.throttle[3];
  motor_msg.m5 = throttles.throttle[4];
  motor_msg.m6 = throttles.throttle[5];
  motor_msg.m7 = throttles.throttle[6];
  motor_msg.m8 = throttles.throttle[7];
  
  pub_motor_.publish(motor_msg);
  count++;
}

void Snapdragon::RosNode::SND::PublishSMCData(const Snapdragon::ControllerManager::controlData& data ){
  snapstack_msgs::SMCData msg;
  msg.header.stamp = ros::Time::now();
  quat2ROS(msg.q_des,data.q_des);
  quat2ROS(msg.q_act,data.q_act);
  quat2ROS(msg.q_err,data.q_err);
  vec2ROS(msg.w_des,data.w_des);
  vec2ROS(msg.w_act,data.w_act);
  vec2ROS(msg.w_err,data.w_err);
  vec2ROS(msg.s,data.s);

  pub_smc_.publish(msg);
}

void Snapdragon::RosNode::SND::BroadcastTF (const Snapdragon::ObserverManager::State& state ){
  geometry_msgs::TransformStamped transformStamped;
 
  float norm = sqrt(pow(state.q.w,2) + pow(state.q.x,2) + pow(state.q.y,2) + pow(state.q.z,2)); 
 
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = vehname_;
  transformStamped.transform.translation.x = state.pos.x;
  transformStamped.transform.translation.y = state.pos.y;
  transformStamped.transform.translation.z = state.pos.z;
  transformStamped.transform.rotation.x = state.q.x/norm;
  transformStamped.transform.rotation.y = state.q.y/norm;
  transformStamped.transform.rotation.z = state.q.z/norm;
  transformStamped.transform.rotation.w = state.q.w/norm;

  br_.sendTransform(transformStamped);
}

void Snapdragon::RosNode::SND::poseCB(const geometry_msgs::PoseStamped& msg) {
  if (observer_man_.calibrated_){
    Vector pos; Quaternion q;
    ROS2vec(pos,msg.pose.position);
    ROS2quat(q,msg.pose.orientation);
    uint64_t timestamp_us = static_cast<uint64_t>(msg.header.stamp.sec*1e6 + msg.header.stamp.nsec*1e-3);
    observer_man_.updateState(observer_man_.state_, pos, q, timestamp_us);
  }
}

void Snapdragon::RosNode::SND::goalCB(const snapstack_msgs::AttitudeCommand& msg){
  if (observer_man_.calibrated_){
    // Update smc command
    desiredAttState newdesState;
    newdesState.power = msg.power;
    ROS2quat(newdesState.q,msg.q);
    ROS2vec(newdesState.w, msg.w);
    ROS2vec(newdesState.F_W, msg.F_W);
    observer_man_.updateSMCState(newdesState);
  }
}

void Snapdragon::RosNode::SND::getParams(Snapdragon::ObserverManager::InitParams& Params, Snapdragon::ControllerManager::InitParams& SmcParams){
  // Filter params
  double theta;
  double kAtt;
  double kGyroBias;
  double kAccelBias;
  double DT;
  double controlDT;
  std::vector<double> polyF_cw, polyF_ccw;
  std::vector<int> motor_spin;
  std::string mixer;
  std::vector<double> Kr;
  std::vector<double> Komega;
  std::vector<double> Jdiag;
  std::vector<double> com; 
  double l, cd;
  
  safeGetParam(nh_, "theta", theta);
  safeGetParam(nh_, "kAtt", kAtt);
  safeGetParam(nh_, "kGyroBias", kGyroBias);
  safeGetParam(nh_, "kAccelBias", kAccelBias);
  safeGetParam(nh_, "controlDT", controlDT);
  safeGetParam(nh_, "thrust_curve_cw", polyF_cw);
  safeGetParam(nh_, "thrust_curve_ccw", polyF_ccw);
  safeGetParam(nh_, "motor_spin", motor_spin);
  safeGetParam(nh_, "cd", cd);
  safeGetParam(nh_, "mixer", mixer);
  safeGetParam(nh_, "broadcast_tf", broadcast_tf_);// should we broadcast a tf?
  safeGetParam(nh_, "sfpro", Params.sfpro);
  safeGetParam(nh_, "Kr", Kr);
  safeGetParam(nh_, "Komega", Komega);
  safeGetParam(nh_, "Jdiag", Jdiag);
  safeGetParam(nh_, "l", l);
  safeGetParam(nh_, "com", com);

  //Smc Params
  SmcParams.Kr = Eigen::Vector3d(Kr[0], Kr[1], Kr[2]);
  SmcParams.Komega = Eigen::Vector3d(Komega[0], Komega[1], Komega[2]);
  SmcParams.J = Eigen::Matrix3d(Eigen::Vector3d(Jdiag[0],  Jdiag[1], Jdiag[2]).asDiagonal()); 
  SmcParams.l = l;
  SmcParams.cd = cd;
  SmcParams.controlDT = controlDT;
  SmcParams.polyF_cw = polyF_cw;
  SmcParams.polyF_ccw = polyF_ccw;
  SmcParams.motor_spin = motor_spin;
  SmcParams.com = com;

  // make case insensitive
  std::transform(mixer.begin(), mixer.end(), mixer.begin(),
    [](unsigned char c){ return std::toupper(c); });
  if (mixer == "QUADX" || mixer == "QUAD_X") {
    SmcParams.mixer = Snapdragon::ControllerManager::Mixer::QUAD_X;
  } else if (mixer == "HEX") {
    SmcParams.mixer = Snapdragon::ControllerManager::Mixer::HEX;
  } else {
    SmcParams.mixer = Snapdragon::ControllerManager::Mixer::INVALID;
    ROS_ERROR("Invalid Mixer");
  }

  ROS_WARN_STREAM((Params.sfpro?"sfpro":"sf") << " board");

  // filter coeffs for a critically damped g-h filter with discount factor theta
  // "Tracking and Kalman Filtering Made Easy" (Eli Brookner, p. 52)
  Params.Kp = 1-std::pow(theta,3);
  Params.Kv = 1.5*(1-std::pow(theta,2))*(1-theta);
  Params.Kq = kAtt;
  Params.Kab = kAccelBias;
  Params.Kgb = kGyroBias;

  // LPF cut-off frequencies
  ros::param::param<double>("~fc/acc_xy", Params.fc_acc_xy, 90.0);
  ros::param::param<double>("~fc/acc_z", Params.fc_acc_z, 90.0);
  ros::param::param<double>("~fc/gyr", Params.fc_gyr, 90.0);

  // Adaptive gyro notch filtering parameters
  ros::param::param<bool>("~anotch/enable", Params.anotch_enable, false);
  ros::param::param<int>("~anotch/nfft", Params.anotch_params.NFFT, 128);
  ros::param::param<int>("~anotch/dual_notch_width_percent", Params.anotch_params.dual_notch_width_percent, 2);
  ros::param::param<int>("~anotch/Q", Params.anotch_params.Q, 360);
  ros::param::param<int>("~anotch/min_hz", Params.anotch_params.min_hz, 60);
  ros::param::param<int>("~anotch/max_hz", Params.anotch_params.max_hz, 200);
}

void Snapdragon::RosNode::SND::vec2ROS(geometry_msgs::Vector3& vros, const Vector& v){
  vros.x = v.x;
  vros.y = v.y;
  vros.z = v.z;
}

void Snapdragon::RosNode::SND::vec2ROS(geometry_msgs::Vector3& vros, const tf2::Vector3& v){
  vros.x = v.x();
  vros.y = v.y();
  vros.z = v.z();
}

void Snapdragon::RosNode::SND::vec2ROS(geometry_msgs::Point& vros, const Vector& v){
  vros.x = v.x;
  vros.y = v.y;
  vros.z = v.z;
}

void Snapdragon::RosNode::SND::ROS2vec(Vector& v, const geometry_msgs::Point& vros){
  v.x = vros.x;
  v.y = vros.y;
  v.z = vros.z;
}

void Snapdragon::RosNode::SND::ROS2vec(Vector& v, const geometry_msgs::Vector3& vros){
  v.x = vros.x;
  v.y = vros.y;
  v.z = vros.z;
}

void Snapdragon::RosNode::SND::quat2ROS(geometry_msgs::Quaternion& qros, const Quaternion& q){
  qros.w = q.w;
  qros.x = q.x;
  qros.y = q.y;
  qros.z = q.z;
}

void Snapdragon::RosNode::SND::ROS2quat(Quaternion& q, const geometry_msgs::Quaternion& qros){
  q.w = qros.w;
  q.x = qros.x;
  q.y = qros.y;
  q.z = qros.z;
}

#ifdef SNAP_APM
void Snapdragon::RosNode::SND::pubApmCB(const ros::TimerEvent& e){
  std_msgs::Float32 msg_vol, msg_curr;
  apm_.read(msg_vol.data, msg_curr.data);

  pub_vol_.publish(msg_vol);
  pub_curr_.publish(msg_curr);
}
#endif
