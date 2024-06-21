/**
 * @file outer_loop_ros.h
 * @brief ROS wrapper for outer loop controller
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 20 August 2020
 */

#include <memory>
#include <sstream>
#include <string>

#include <ros/ros.h>
#include <outer_loop/OuterLoopConfig.h>
#include <dynamic_reconfigure/server.h>
#include <tf2/utils.h>
#include <Eigen/Dense>

#include <eigen_conversions/eigen_msg.h>

#include <snapstack_msgs/State.h>
#include <snapstack_msgs/Goal.h>
#include <snapstack_msgs/AttitudeCommand.h>
#include <snapstack_msgs/ControlLog.h>

#include <outer_loop/outer_loop.h>

namespace acl {
namespace outer_loop {

  class OuterLoopROS
  {
  public:
    OuterLoopROS(const ros::NodeHandle& nh, const ros::NodeHandle& nhp);
    ~OuterLoopROS();

  private:
    ros::NodeHandle nh_, nhp_;
    ros::Subscriber sub_state_, sub_goal_;
    ros::Publisher pub_att_cmd_, pub_log_;
    ros::Timer tim_cntrl_;
    dynamic_reconfigure::Server<::outer_loop::OuterLoopConfig> server_;

    /// \brief Data members
    enum class Mode { Preflight, SpinningUp, Flying, EmergencyStop };
    Mode mode_ = Mode::Preflight; ///< current flight sequence state
    ros::Time t_start_; ///< ros::Time that spin up was initialized
    snapstack_msgs::Goal goalmsg_; ///< most recent goal received
    Goal goal_;
    snapstack_msgs::State statemsg_; ///< most recent state received
    State state_;
    std::string safetyKillReason_; ///< human reason for cutting motors
    std::unique_ptr<OuterLoop> olcntrl_;

    /// \brief Parameters
    std::string name_; ///< vehicle name
    double control_dt_; ///< send attitude cmds at this period
    double Tspinup_; ///< motor spinup time to avoid nonlinear motor regime
    double spinup_thrust_gs_; ///< Normalized thrust [g's] to use during spinup
    double spinup_thrust_; ///< Amount of thrust [N] to use during spinup
    double alt_limit_; ///< If altitude rises above this, motor power is cut

    /// \brief ROS callbacks
    void cntrlCb(const ros::TimerEvent& e);
    void stateCb(const snapstack_msgs::State& msg);
    void goalCb(const snapstack_msgs::Goal& msg);
    void dynamicReconfigurePidParamsCb(::outer_loop::OuterLoopConfig& cfg, uint32_t level);

    /// \brief Helper methods
    void publishLog(const snapstack_msgs::AttitudeCommand& attmsg);
    void screenPrint(const snapstack_msgs::AttitudeCommand& attmsg);

    /// \brief Preflight Checks
    bool doPreflightChecksPass();
    bool checkState();

    /// \brief In-flight Saftey Checks
    bool doSafetyChecksPass();
    bool checkAltitude();
    bool checkComms();
  };

} // ns outer_loop
} // ns acl
