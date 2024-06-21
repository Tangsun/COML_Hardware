#include <ros/ros.h>

#include <std_msgs/Duration.h>
#include <geometry_msgs/PoseStamped.h>

#include <snapstack_msgs/CommAge.h>
#include <snapstack_msgs/Goal.h>

namespace acl {
namespace comm_monitor {

class CommMonitorNode{
public:
  ros::NodeHandle nh_, nhp_;
  ros::Publisher pub_vicon_;
  ros::Publisher pub_goal_;
  ros::Publisher pub_ages_;
  ros::Subscriber sub_vicon_pose_;
  ros::Subscriber sub_goal_;
  ros::Timer comm_monitor_clock_;
  ros::Time latest_vicon_odroid_time_;
  ros::Time latest_goal_odroid_time_;
  geometry_msgs::PoseStamped latest_vicon_;
  snapstack_msgs::Goal latest_goal_;

  bool has_vicon;
  bool has_goal;

  CommMonitorNode(const ros::NodeHandle& nh, const ros::NodeHandle& nhp)
  : nh_(nh), nhp_(nhp)
  {
    // Read in parameters
    int comm_monitor_hz;
    nhp_.param<int>("comm_monitor_hz", comm_monitor_hz, 100);
    float comm_monitor_period = 1.0/comm_monitor_hz;

    // Initialize    
    pub_ages_ = nh_.advertise<snapstack_msgs::CommAge>("comm_ages",1);

    sub_vicon_pose_ = nh_.subscribe("world",1,&CommMonitorNode::newViconReceivedCallback,this);
    sub_goal_ = nh_.subscribe("goal",1,&CommMonitorNode::newGoalReceivedCallback,this);

    comm_monitor_clock_ = nhp_.createTimer(ros::Duration(comm_monitor_period),&CommMonitorNode::commMonitorCycle,this);

  }
  ~CommMonitorNode(){}

  void newViconReceivedCallback(const geometry_msgs::PoseStamped& vicon)
  {    
    if (vicon.header.stamp > latest_vicon_.header.stamp){
      latest_vicon_odroid_time_ = ros::Time::now();
      // ROS_INFO("Got new vicon.");
      latest_vicon_ = vicon;
    }
  }
  
  void newGoalReceivedCallback(const snapstack_msgs::Goal& goal)
  {
    if (goal.header.stamp > latest_goal_.header.stamp){
      latest_goal_odroid_time_ = ros::Time::now();
      // ROS_INFO("Got new goal.");
      latest_goal_ = goal;
    }
  }

  float checkViconAge()
  {
    float time_since_last_vicon;
    time_since_last_vicon = (ros::Time::now()-latest_vicon_odroid_time_).toSec();
    return time_since_last_vicon;
  }
  float checkGoalAge()
  {
    float time_since_last_goal;
    time_since_last_goal = (ros::Time::now()-latest_goal_odroid_time_).toSec();
    return time_since_last_goal;
  }

  snapstack_msgs::CommAge packageAgesIntoMessage(float time_since_last_vicon, float time_since_last_goal)
  {
    snapstack_msgs::CommAge ages_message;
    //TODO: Add frame_id?
    ages_message.header.stamp = ros::Time::now();
    ages_message.vicon_age_secs = time_since_last_vicon;
    ages_message.goal_age_secs = time_since_last_goal;
    return ages_message;
  }

  void commMonitorCycle(const ros::TimerEvent& event)
  {
    float time_since_last_vicon = checkViconAge();
    float time_since_last_goal = checkGoalAge();
    snapstack_msgs::CommAge ages = packageAgesIntoMessage(time_since_last_vicon,time_since_last_goal);
    // Only publish if have recevied at least one msg
    // Eventually might need to add in the goal age condition latest_goal_.header.stamp > ros::Time(0))) { 
    if ((latest_vicon_.header.stamp > ros::Time(0))){
      pub_ages_.publish(ages);
    }
  }
};

} // ns comm_monitor
} // ns acl


int main(int argc, char **argv)
{
  ros::init(argc, argv, "comm_monitor");
  ros::NodeHandle nhtopics("");
  ros::NodeHandle nhparams("~");
  acl::comm_monitor::CommMonitorNode node(nhtopics, nhparams);
  ros::spin();
  return 0;
}
