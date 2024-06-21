/**
 * @file snap_sim.cpp
 * @brief ROS wrapper for ACL multirotor dynamic simulation
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 23 December 2019
 */

#include "snap_sim/snap_sim.h"


namespace acl {
namespace snap_sim {

SnapSim::SnapSim(const ros::NodeHandle& nh, const ros::NodeHandle& nhp)
  : nh_(nh), nhp_(nhp), motorcmds_({0})
{
  // Check namespace to find name of vehicle
  name_ = ros::this_node::getNamespace();
  size_t n = name_.find_first_not_of('/');
  name_.erase(0, n); // remove leading slashes
  if (name_.empty()) {
    ROS_ERROR("Error :: You should be using a launch file to specify the "
              "node namespace!\n");
    ros::shutdown();
    return;
  }

  init();

  //
  // ROS timers
  //

  nhp_.param<double>("imu_dt", imu_dt_, 0.002);
  nhp_.param<double>("mocap_dt", mocap_dt_, 0.01);
  time_last_simStepCb_ = ros::Time::now();
  tim_imu_ = nh_.createTimer(ros::Duration(imu_dt_), &SnapSim::simStepCb, this);
  tim_mocap_ = nh_.createTimer(ros::Duration(mocap_dt_), &SnapSim::mocapCb, this);
  wind_ =  nh_.subscribe("wind", 1, &SnapSim::windCb, this);
}

// ----------------------------------------------------------------------------

SnapSim::~SnapSim()
{
  if (escthread_.joinable()) escthread_.join();
  if (armthread_.joinable()) armthread_.join();
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void SnapSim::init()
{
  //
  // Simulated mocap initialization
  //

  // what frame should mocap be published w.r.t?
  nhp_.param<std::string>("mocap_tf_parent_frame", mocap_parent_frame_, "world");
  nhp_.param<bool>("mocap_tf_broadcast", mocap_tf_broadcast_, true);

  pub_pose_ = nh_.advertise<geometry_msgs::PoseStamped>(mocap_parent_frame_, 1);
  pub_twist_ = nh_.advertise<geometry_msgs::TwistStamped>("mocap/twist", 1);

  //
  // Visualization
  //

  nhp_.param<bool>("viz_mesh", viz_mesh_, true);
  nhp_.param<std::string>("mesh_uri", mesh_uri_, "package://snap_sim/meshes/hexarotor.dae");
  nhp_.param<double>("mesh_scale", mesh_scale_, 0.75);
  if (viz_mesh_) pub_vizmesh_ = nh_.advertise<visualization_msgs::Marker>("viz_mesh", 1);

  //
  // Build a simulated multirotor
  //

  // initialize state
  RigidBody::State state;
  nhp_.param<double>("init/x", state.p.x(), 0.0);
  nhp_.param<double>("init/y", state.p.y(), 0.0);
  nhp_.param<double>("init/z", state.p.z(), 0.0);
  nhp_.param<double>("init/q_x", state.q.x(), 0.0);
  nhp_.param<double>("init/q_y", state.q.y(), 0.0);
  nhp_.param<double>("init/q_z", state.q.z(), 0.0);
  nhp_.param<double>("init/q_w", state.q.w(), 1.0);
  //std::cout << "state.q.w: " <<  state.q.w() << std::endl;
  const double eps = 1e-2;
  if (state.q.norm() > eps){
    state.q.normalize();
  } else {
    ROS_ERROR("Init attitude quaternion has norm() < %f. Using identity quaternion.", eps);
    state.q = Eigen::Quaterniond::Identity();
  } 
  state.v = Eigen::Vector3d::Zero();
  state.w = Eigen::Vector3d::Zero();

  // physical vehicle parameters
  Multirotor::Params params;
  nhp_.param<double>("mass", params.mass, 1.0);
  params.J = RigidBody::Inertia::Zero();
  nhp_.param<double>("Jx", params.J(0,0), 0.01);
  nhp_.param<double>("Jy", params.J(1,1), 0.01);
  nhp_.param<double>("Jz", params.J(2,2), 0.01);
  nhp_.param<double>("linvel_drag_lin", params.muVLin, 0.0);
  nhp_.param<double>("linvel_drag_quad", params.muVQuad, 0.0);
  nhp_.param<double>("angvel_drag", params.muOmega, 0.0);
  nhp_.getParam("ground_effect", params.polyGroundEffect);

  // propulsion properties
  Multirotor::Rotor rotor, rotor_cw, rotor_ccw;
  nhp_.getParam("inverse_thrust_curve_cw", rotor_cw.polyF);
  nhp_.getParam("inverse_thrust_curve_ccw", rotor_ccw.polyF);
  nhp_.getParam("torque_curve_cw", rotor_cw.polyT);
  nhp_.getParam("torque_curve_ccw", rotor_ccw.polyT);
  nhp_.param<double>("tau_spin_up", rotor.tauUp, 0.0);
  nhp_.param<double>("tau_spin_down", rotor.tauDn, 0.0);
  nhp_.param<double>("min_thrust", rotor.minF, 0.0);
  nhp_.param<double>("max_thrust", rotor.maxF, 0.0);
  nhp_.param<double>("min_torque", rotor.minT, 0.0);
  nhp_.param<double>("max_torque", rotor.maxT, 0.0);

  // center of mass
  std::vector<double> com;
  nhp_.getParam("com", com);
  params.center_of_mass << com[0], com[1], com[2];

  // motor layout
  std::vector<double> motor_positions;
  std::vector<double> motor_directions;
  std::vector<int> motor_spin;
  nhp_.getParam("motor_positions", motor_positions);
  nhp_.getParam("motor_directions", motor_directions);
  nhp_.getParam("motor_spin", motor_spin);
  params.motors.resize(motor_spin.size());
  for (size_t i=0; i<params.motors.size(); ++i) {
    auto& motor = params.motors[i];

    // assumption: motors differ only in thrust/torque curve according to spinning direction. 
    motor.rotor = rotor;
    if (motor_spin[i] < 0){
      // Clockwise spinning rotor
      motor.rotor.polyF = rotor_cw.polyF;
      motor.rotor.polyT = rotor_cw.polyT;
    }else{
      // Counter-clockwise spinning rotor
      motor.rotor.polyF = rotor_ccw.polyF;
      motor.rotor.polyT = rotor_ccw.polyT;
    }

    // which direction do the propellers spin?
    motor.spin = motor_spin[i];

    // set motor position and thrust direction
    for (size_t j=0; j<3; ++j) {
      motor.position(j) = motor_positions[3*i + j];
      motor.direction(j) = motor_directions[3*i + j];
    }
  }

  // sensor parameters
  nhp_.param<double>("gyro/stdev", params.gyro_stdev, 0.0);
  nhp_.param<double>("gyro/bias_walk_stdev", params.gyro_walk_stdev, 0.0);
  nhp_.param<double>("accel/stdev", params.accel_stdev, 0.0);
  nhp_.param<double>("accel/bias_walk_stdev", params.accel_walk_stdev, 0.0);

  // wind parameters
  nhp_.param<double>("wx_nominal", params.wind_nominal.x(), 0.0);
  nhp_.param<double>("wy_nominal", params.wind_nominal.y(), 0.0);
  nhp_.param<double>("wz_nominal", params.wind_nominal.z(), 0.0);
  nhp_.param<double>("wx_gust_bound", params.wind_gust_bound.x(), 0.0);
  nhp_.param<double>("wy_gust_bound", params.wind_gust_bound.y(), 0.0);
  nhp_.param<double>("wz_gust_bound", params.wind_gust_bound.z(), 0.0);

  multirotor_.reset(new Multirotor(state, params));
  physics_.addRigidBody(multirotor_);

  //
  // SIL Communications
  //

  const size_t imukey = acl::ipc::createKeyFromStr(name_, "imu");
  const size_t esckey = acl::ipc::createKeyFromStr(name_, "esc");

  // unique key is used to access the same shmem location
  imuserver_.reset(new acl::ipc::Server<sensor_imu>(imukey));
  escclient_.reset(new acl::ipc::Client<esc_commands>(esckey));

  //
  // ESC Commands thread
  //

  escthread_ = std::thread(&SnapSim::escReadThread, this);

  //
  // Auto-arming
  //

  bool autoarm;
  nhp_.param<bool>("autoarm", autoarm, true);

  // only start this thread if auto arming is requested
  if (autoarm) armthread_ = std::thread(&SnapSim::armThread, this);
}

// ----------------------------------------------------------------------------

void SnapSim::armThread()
{
  auto srv_arm = nh_.serviceClient<std_srvs::SetBool>("snap/arm");

  // wait until snap stack has been started
  srv_arm.waitForExistence();

  // create request to arm
  std_srvs::SetBool srv;
  srv.request.data = true;

  // slowly request to arm until we can arm. Multiple tries are necessary
  // since arming is not allowed until the IMU has been calibrated.
  ros::Rate r(2);
  while (ros::ok() && !srv.response.success) {
    srv_arm.call(srv);
    r.sleep();
  }
}

// ----------------------------------------------------------------------------

void SnapSim::escReadThread()
{
  static constexpr uint16_t PWM_MAX = acl::ESCInterface::PWM_MAX_PULSE_WIDTH;
  static constexpr uint16_t PWM_MIN = acl::ESCInterface::PWM_MIN_PULSE_WIDTH;

  while (ros::ok()) {
    esc_commands esccmds;
    bool rcvd = escclient_->read(&esccmds);

    std::lock_guard<std::mutex> lck(escmtx_);
    for (size_t i=0; i<Multirotor::NUM_PWM; ++i)
      motorcmds_[i] = static_cast<double>(esccmds.pwm[i] - PWM_MIN) / (PWM_MAX - PWM_MIN);
  }
}

// ----------------------------------------------------------------------------

void SnapSim::simStepCb(const ros::TimerEvent& e)
{
  {
    // set the currently desired pwm motor commands
    std::lock_guard<std::mutex> lck(escmtx_);
    multirotor_->setMotorCommands(motorcmds_);
  }

  // instead of letting the physics world handle time, we just always use ROS.
  ros::Time now = ros::Time::now();
  physics_.setSimTime(now.toSec());

  // simulate forward
  double dt = (now - time_last_simStepCb_).toSec();
  if(dt <= 0)
    return;

  physics_.step(dt);
  time_last_simStepCb_ = now;

  // Read sensors
  Eigen::Vector3d acc, gyr;
  double imu_time = multirotor_->readIMU(acc, gyr);

  imu_.timestamp_in_us = static_cast<uint64_t>(imu_time*1e6);
  imu_.sequence_number++;
  // Rotate the measurements so that they correspond to the IMU orientation
  // of the sf board (eagle8074). Further, note that IMU measures in g's.
  imu_.linear_acceleration[0] =   acc.x() / GRAVITY.norm();
  imu_.linear_acceleration[1] = - acc.y() / GRAVITY.norm();
  imu_.linear_acceleration[2] = - acc.z() / GRAVITY.norm();
  imu_.angular_velocity[0] =   gyr.x();
  imu_.angular_velocity[1] = - gyr.y();
  imu_.angular_velocity[2] = - gyr.z();
  imuserver_->send(imu_);
}

// ----------------------------------------------------------------------------

void SnapSim::mocapCb(const ros::TimerEvent& e)
{
  ros::Time timestamp = ros::Time::now();

  geometry_msgs::PoseStamped msgpose;
  msgpose.header.stamp = timestamp;
  msgpose.header.frame_id = mocap_parent_frame_;
  msgpose.pose.position.x = multirotor_->getState().p.x();
  msgpose.pose.position.y = multirotor_->getState().p.y();
  msgpose.pose.position.z = multirotor_->getState().p.z();
  msgpose.pose.orientation.w = multirotor_->getState().q.w();
  msgpose.pose.orientation.x = multirotor_->getState().q.x();
  msgpose.pose.orientation.y = multirotor_->getState().q.y();
  msgpose.pose.orientation.z = multirotor_->getState().q.z();
  pub_pose_.publish(msgpose);

  geometry_msgs::TwistStamped msgtwist;
  msgtwist.header.stamp = timestamp;
  msgtwist.header.frame_id = mocap_parent_frame_; // expressed in world frame
  msgtwist.twist.linear.x = multirotor_->getState().v.x();
  msgtwist.twist.linear.y = multirotor_->getState().v.y();
  msgtwist.twist.linear.z = multirotor_->getState().v.z();
  msgtwist.twist.angular.x = multirotor_->getState().w.x();
  msgtwist.twist.angular.y = multirotor_->getState().w.y();
  msgtwist.twist.angular.z = multirotor_->getState().w.z();
  pub_twist_.publish(msgtwist);

  if (mocap_tf_broadcast_) {
    geometry_msgs::TransformStamped msgtf;
    msgtf.header = msgpose.header;
    msgtf.transform.rotation = msgpose.pose.orientation;
    msgtf.transform.translation.x = msgpose.pose.position.x;
    msgtf.transform.translation.y = msgpose.pose.position.y;
    msgtf.transform.translation.z = msgpose.pose.position.z;
    msgtf.child_frame_id = name_;
    br_.sendTransform(msgtf);
  }

  //
  // Visualization
  //

  if (viz_mesh_) {
    visualization_msgs::Marker m;
    m.action = visualization_msgs::Marker::ADD;
    m.ns = "mesh";
    m.id = 1;
    m.type = visualization_msgs::Marker::MESH_RESOURCE;
    m.mesh_resource = mesh_uri_;
    m.mesh_use_embedded_materials = true;
    m.color.a = 0.0;
    m.color.r = 0.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    m.header = msgpose.header;
    m.pose = msgpose.pose;
    m.scale.x = mesh_scale_;
    m.scale.y = mesh_scale_;
    m.scale.z = mesh_scale_;
    m.lifetime = ros::Duration(1.0);

    pub_vizmesh_.publish(m);
  }
}

void SnapSim::windCb(const snapstack_msgs::Wind& msg)
{
  // ROS_INFO("Received wind speed: [%f], direction: [%f]", msg.w_nominal.x, msg.w_gust.y);
  Eigen::Vector3d wind = Eigen::Vector3d(msg.w_nominal.x,  msg.w_nominal.y,  msg.w_nominal.z);  
  multirotor_->setWind(wind); 
}

} // ns snap_sim
} // ns acl
