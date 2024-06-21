/**
 * @file outer_loop_ros.cpp
 * @brief ROS wrapper for outer loop controller
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 20 August 2020
 */

#include <outer_loop/outer_loop_ros.h>

namespace acl {
namespace outer_loop {

OuterLoopROS::OuterLoopROS(const ros::NodeHandle& nh, const ros::NodeHandle& nhp)
  : nh_(nh), nhp_(nhp)
{
  // Check namespace to find name of vehicle
  name_ = ros::this_node::getNamespace();
  size_t n = name_.find_first_not_of('/');
  name_.erase(0, n); // remove leading slashes
  if (name_.empty()) {
    ROS_ERROR("Error :: Vehicle namespace is missing. Hint: use launch file "
              "with node namespacing\n");
    ros::shutdown();
    return;
  }

  //
  // Load ROS parameters
  //

  OuterLoop::Parameters p;
  double kp_xy, ki_xy, kd_xy, kp_z, ki_z, kd_z;
  double maxPosErr_xy, maxPosErr_z, maxVelErr_xy, maxVelErr_z;

  nhp_.param<double>("control_dt", control_dt_, 0.01);
  nhp_.param<double>("spinup/time", Tspinup_, 1.0);
  nhp_.param<double>("spinup/thrust_gs", spinup_thrust_gs_, 0.5);
  nhp_.param<double>("mass", p.mass, 1.0);
  nhp_.param<double>("Kp_xy", kp_xy, 1.0);
  nhp_.param<double>("Ki_xy", ki_xy, 0.0);
  nhp_.param<double>("Kd_xy", kd_xy, 0.0);
  nhp_.param<double>("Kp_z", kp_z, 1.0);
  nhp_.param<double>("Ki_z", ki_z, 0.0);
  nhp_.param<double>("Kd_z", kd_z, 0.0);
  nhp_.param<double>("maxPosErr/xy", maxPosErr_xy, 1.0);
  nhp_.param<double>("maxPosErr/z", maxPosErr_z, 1.0);
  nhp_.param<double>("maxVelErr/xy", maxVelErr_xy, 1.0);
  nhp_.param<double>("maxVelErr/z", maxVelErr_z, 1.0);
  nhp_.param<double>("safety/alt_limit", alt_limit_, 6.0);

  // make sure that spinup_thrust is safe
  if (spinup_thrust_gs_ > 0.9) {
    ROS_WARN_STREAM("You requested a spinup/thrust_gs of " << spinup_thrust_gs_ << "."
                    << std::endl << std::endl
                    << "This value is used to calculate the idling throttle of the "
                    << "motors before takeoff and should be just enough to make them "
                    << "spin (getting out of the nonlinearity of motors turning on) "
                    << "but not enough to overcome the force of gravity, F = mg."
                    << std::endl << std::endl
                    << "Overriding spinup/thrust_gs with a safer value, 0.5");
    spinup_thrust_gs_ = 0.5;
  }
  // calculate thrust in [N] to use
  spinup_thrust_ = spinup_thrust_gs_ * p.mass * 9.81;

  p.Kp << kp_xy, kp_xy, kp_z;
  p.Ki << ki_xy, ki_xy, ki_z;
  p.Kd << kd_xy, kd_xy, kd_z;
  p.maxPosErr << maxPosErr_xy, maxPosErr_xy, maxPosErr_z;
  p.maxVelErr << maxVelErr_xy, maxVelErr_xy, maxVelErr_z;

  //
  // Initialize trajectory tracking controller
  //

  olcntrl_.reset(new OuterLoop(p));

  //
  // ROS timers
  //

  tim_cntrl_ = nhp_.createTimer(ros::Duration(control_dt_), &OuterLoopROS::cntrlCb, this);

  //
  // ROS pub/sub
  //

  sub_state_ = nh_.subscribe("state", 1, &OuterLoopROS::stateCb, this);
  sub_goal_ = nh_.subscribe("goal", 1, &OuterLoopROS::goalCb, this);
  // sub_comm_ages_ = nh_.subscribe("comm_ages", 1, &OuterLoopROS::commagesCb, this);

  pub_att_cmd_ = nh_.advertise<snapstack_msgs::AttitudeCommand>("attcmd", 1);
  pub_log_ = nh_.advertise<snapstack_msgs::ControlLog>("log", 1);

  // 
  // ROS dynamic reconfigure
  //
  dynamic_reconfigure::Server<::outer_loop::OuterLoopConfig>::CallbackType f;
  f = boost::bind(&OuterLoopROS::dynamicReconfigurePidParamsCb, this, _1, _2);
  server_.setCallback(f);
}

// ----------------------------------------------------------------------------

OuterLoopROS::~OuterLoopROS()
{
  // TODO: send cut power message <-- does this work here?
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void OuterLoopROS::dynamicReconfigurePidParamsCb(::outer_loop::OuterLoopConfig& cfg, uint32_t level)
{
  ROS_INFO_STREAM("Reconfigure request received.\n");
  olcntrl_->setHorizontalPositionPidParams(cfg.Kp_xy, cfg.Ki_xy, cfg.Kd_xy);
  olcntrl_->setAltitudePidParams(cfg.Kp_z, cfg.Ki_z, cfg.Kd_z);
}

void OuterLoopROS::cntrlCb(const ros::TimerEvent& e)
{

  AttCmd cmd;
  cmd.q = state_.q;
  cmd.w = Eigen::Vector3d::Zero();
  cmd.F_W = Eigen::Vector3d::Zero();

  const ros::Time t_now = ros::Time::now();
  if (t_now.isZero()) {
    // Wait for valid time
    return;
  }

  // if high-level planner doesn't allow power, we won't fly
  if (!goalmsg_.power) {
    // however, even if a flight is requested, if we ever had to emergency
    // stop, motors will not be allowed to spin until after the node restarts.
    if (mode_ != Mode::EmergencyStop) mode_ = Mode::Preflight;
  }

  // passthrough the current state and goal to keep log up to date.
  // n.b., this clears outer loop control internally logged signals,
  // so this should be called before computeAttitudeCommand.
  olcntrl_->updateLog(state_);

  //
  // Flight Sequence State Machine
  //

  if (mode_ == Mode::Preflight) {

    if (goalmsg_.power) {
      mode_ = Mode::SpinningUp;
      t_start_ = t_now;
    }

    if (!doPreflightChecksPass()) {
      mode_ = Mode::Preflight;
    }

  }

  if (mode_ == Mode::SpinningUp) {

    if (t_now < t_start_ + ros::Duration(Tspinup_)) {
      cmd.q = state_.q;
      cmd.w = Eigen::Vector3d::Zero();
      cmd.F_W = spinup_thrust_ * Eigen::Vector3d::UnitZ();
    } else {
      olcntrl_->reset();
      mode_ = Mode::Flying;
    }

    ROS_WARN_THROTTLE(0.5, "Spinning up motors.");
  }

  if (mode_ == Mode::Flying) {
    cmd = olcntrl_->computeAttitudeCommand(t_now.toSec(), state_, goal_);

    // safety checks?
    if (!doSafetyChecksPass()) {
      mode_ = Mode::EmergencyStop;
    }
  }

  if (mode_ == Mode::EmergencyStop) {
    cmd.q = state_.q;
    cmd.w = Eigen::Vector3d::Zero();
    cmd.F_W = Eigen::Vector3d::Zero();

    ROS_WARN_THROTTLE(0.5, "Emergency stop.");
  }

  //
  // Publish command via ROS
  //

  snapstack_msgs::AttitudeCommand attmsg;
  attmsg.header.stamp = t_now;
  attmsg.power = (mode_ == Mode::SpinningUp || mode_ == Mode::Flying);
  tf::quaternionEigenToMsg(cmd.q, attmsg.q);
  tf::vectorEigenToMsg(cmd.w, attmsg.w);
  tf::vectorEigenToMsg(cmd.F_W, attmsg.F_W);
  pub_att_cmd_.publish(attmsg);

  publishLog(attmsg);
  screenPrint(attmsg);

  // screen print
}

// ----------------------------------------------------------------------------

void OuterLoopROS::stateCb(const snapstack_msgs::State& msg)
{
  statemsg_ = msg;
  state_.t = msg.header.stamp.toSec();
  tf::pointMsgToEigen(msg.pos, state_.p);
  tf::vectorMsgToEigen(msg.vel, state_.v);
  tf::quaternionMsgToEigen(msg.quat, state_.q);
  tf::vectorMsgToEigen(msg.w, state_.w);
}

// ----------------------------------------------------------------------------

void OuterLoopROS::goalCb(const snapstack_msgs::Goal& msg)
{
  goalmsg_ = msg;
  goal_.t = msg.header.stamp.toSec();
  tf::pointMsgToEigen(msg.p, goal_.p);
  tf::vectorMsgToEigen(msg.v, goal_.v);
  tf::vectorMsgToEigen(msg.a, goal_.a);
  tf::vectorMsgToEigen(msg.j, goal_.j);
  goal_.psi = msg.psi;
  goal_.dpsi = msg.dpsi;
  goal_.mode_xy = static_cast<Goal::Mode>(msg.mode_xy);
  goal_.mode_z = static_cast<Goal::Mode>(msg.mode_z);
}

// ----------------------------------------------------------------------------

void OuterLoopROS::publishLog(const snapstack_msgs::AttitudeCommand& attmsg)
{
  const auto& log = olcntrl_->getLog();

  snapstack_msgs::ControlLog msg;
  msg.header = attmsg.header;

  tf::vectorEigenToMsg(log.p, msg.p);
  tf::vectorEigenToMsg(log.p_ref, msg.p_ref);
  tf::vectorEigenToMsg(log.p_err, msg.p_err);
  tf::vectorEigenToMsg(log.p_err_int, msg.p_err_int);

  tf::vectorEigenToMsg(log.v, msg.v);
  tf::vectorEigenToMsg(log.v_ref, msg.v_ref);
  tf::vectorEigenToMsg(log.v_err, msg.v_err);

  tf::vectorEigenToMsg(log.a_ff, msg.a_ff);
  tf::vectorEigenToMsg(log.a_fb, msg.a_fb);

  tf::vectorEigenToMsg(log.j_ff, msg.j_ff);
  tf::vectorEigenToMsg(log.j_fb, msg.j_fb);

  tf::quaternionEigenToMsg(log.q, msg.q);
  tf::quaternionEigenToMsg(log.q_ref, msg.q_ref);
  {
    double R, P, Y;
    tf2::Quaternion q(log.q.x(), log.q.y(), log.q.z(), log.q.w());
    tf2::Matrix3x3(q).getRPY(R, P, Y);
    msg.rpy.x = R;
    msg.rpy.y = P;
    msg.rpy.z = Y;
  }
  {
    double R, P, Y;
    tf2::Quaternion q(log.q_ref.x(), log.q_ref.y(), log.q_ref.z(), log.q_ref.w());
    tf2::Matrix3x3(q).getRPY(R, P, Y);
    msg.rpy_ref.x = R;
    msg.rpy_ref.y = P;
    msg.rpy_ref.z = Y;
  }

  tf::vectorEigenToMsg(log.w, msg.w);
  tf::vectorEigenToMsg(log.w_ref, msg.w_ref);

  tf::vectorEigenToMsg(log.F_W, msg.F_W);

   msg.power = attmsg.power;

  pub_log_.publish(msg);
}

// ----------------------------------------------------------------------------

void OuterLoopROS::screenPrint(const snapstack_msgs::AttitudeCommand& attmsg)
{
  const auto& log = olcntrl_->getLog();

  std::ostringstream str;
  str.setf(std::ios::fixed); // give all the doubles the same precision
  str.setf(std::ios::showpos); // show +/- signs always
  str << std::setprecision(4) << std::endl;

  str << "Act Pos:  x: " << log.p.x() << "  y: " << log.p.y()
      << "  z: " << log.p.z() << std::endl;
  str << "Des Pos:  x: " << log.p_ref.x() << "  y: " << log.p_ref.y()
      << "  z: " << log.p_ref.z() << std::endl;
  str << std::endl;

  str << "Act Vel:  x: " << log.v.x() << "  y: " << log.v.y()
      << "  z: " << log.v.z() << std::endl;
  str << "Des Vel:  x: " << log.v_ref.x() << "  y: " << log.v_ref.y()
      << "  z: " << log.v_ref.z() << std::endl;
  str << std::endl;

  double R, P, Y;
  {
    tf2::Quaternion q(log.q.x(), log.q.y(), log.q.z(), log.q.w());
    tf2::Matrix3x3(q).getRPY(R, P, Y);
  }
  str << "Act Att:  r: " << R << "  p: " << P << "  y: " << Y << std::endl;
  {
    tf2::Quaternion q(log.q_ref.x(), log.q_ref.y(), log.q_ref.z(), log.q_ref.w());
    tf2::Matrix3x3(q).getRPY(R, P, Y);
  }
  str << "Des Att:  r: " << R << "  p: " << P << "  y: " << Y << std::endl;
  str << std::endl;

  str << "Act Rate: p: " << log.w.x() << "  q: " << log.w.y()
      << "  r: " << log.w.z() << std::endl;
  str << "Des Rate: p: " << log.w_ref.x() << "  q: " << log.w_ref.y()
      << "  r: " << log.w_ref.z() << std::endl;
  str << std::endl;

  std::string power = (attmsg.power) ? "\033[1;31mY\033[0m" : "\033[1;33mN\033[0m";
  std::string mode_xy = "POS", mode_z = "POS";
  if (log.mode_xy == Goal::Mode::VEL_CTRL) mode_xy = "VEL";
  else if (log.mode_xy == Goal::Mode::ACC_CTRL) mode_xy = "ACC";
  if (log.mode_z == Goal::Mode::VEL_CTRL) mode_z = "VEL";
  else if (log.mode_z == Goal::Mode::ACC_CTRL) mode_z = "ACC";
  str << "Motors On: " << power << "  Thrust: "
      << ((log.F_W.norm()>0)?"\033[97m":"") << log.F_W.norm() << "\033[0m"
      << std::endl;

  str << "XY-Mode: " << mode_xy
      << "  Z-Mode: " << mode_z << std::endl;
  str << std::endl;

  str << "Seconds since last state update: "
      << (ros::Time::now() - statemsg_.header.stamp).toSec() << std::endl;
  if (mode_ == Mode::EmergencyStop) {
    str << "Safety motor kill: \033[1;31m"
        << safetyKillReason_ << "\033[0m" << std::endl;
  }
  str << std::endl;

  ROS_INFO_STREAM_THROTTLE(0.5, str.str());
}

// ----------------------------------------------------------------------------
// Preflight Checks
// ----------------------------------------------------------------------------

bool OuterLoopROS::doPreflightChecksPass()
{
  return checkState();
}

// ----------------------------------------------------------------------------

bool OuterLoopROS::checkState()
{
  if (state_.t == -1) {
    ROS_WARN_THROTTLE(0.5,
              "Preflight checks --- waiting on state data from autopilot. "
                                   "Is IMU calibration complete?");
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
// In-flight Safety Checks
// ----------------------------------------------------------------------------

bool OuterLoopROS::doSafetyChecksPass()
{
  return checkAltitude() && checkComms();
}

// ----------------------------------------------------------------------------

bool OuterLoopROS::checkAltitude()
{
  if (state_.p.z() > alt_limit_) {
    ROS_ERROR_STREAM("Safety --- Altitude check failed (" << state_.p.z()
                      << " > " << alt_limit_ << ").");
    safetyKillReason_ += "ALT ";
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------

bool OuterLoopROS::checkComms()
{
  if (false) {
    safetyKillReason_ += "COMM ";
  }
  return true;
}

} // ns outer_loop
} // ns acl
