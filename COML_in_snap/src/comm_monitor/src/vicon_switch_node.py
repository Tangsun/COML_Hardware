#!/usr/bin/env python
import rospy
from copy import deepcopy
from geometry_msgs.msg import PoseStamped, TwistStamped
from acl_msgs.msg import ViconState, QuadGoal
from collections import deque
from std_srvs.srv import Empty,EmptyRequest,EmptyResponse

class ViconSwitchNode():
	def __init__(self):
		self.node_name = rospy.get_name()

		self.publish_vicon_flag = True
		self.publish_goal_flag = True

		self.pub_vicon = rospy.Publisher("~vicon_out",ViconState,queue_size=1)
		self.pub_goal = rospy.Publisher("~goal_out",QuadGoal,queue_size=1)

		self.sub_vicon = rospy.Subscriber("~vicon_in",ViconState,self.cbVicon,queue_size=1)
		self.sub_goal = rospy.Subscriber("~goal_in",QuadGoal,self.cbGoal,queue_size=1)

		self.srv_pass = rospy.Service("~pass", Empty, self.cbSrvPass)
		self.srv_cut = rospy.Service("~cut", Empty, self.cbSrvCut)
		self.srv_pass_vicon = rospy.Service("~pass_vicon", Empty, self.cbSrvPassVicon)
		self.srv_cut_vicon = rospy.Service("~cut_vicon", Empty, self.cbSrvCutVicon)
		self.srv_pass_goal = rospy.Service("~pass_goal", Empty, self.cbSrvPassGoal)
		self.srv_cut_goal = rospy.Service("~cut_goal", Empty, self.cbSrvCutGoal)
		self.srv_cut_timed = rospy.Service("~cut_timed", Empty, self.cbSrvCutTimed)
		self.cbSrvPass(EmptyRequest())

	def cbVicon(self,msg):
		if self.publish_vicon_flag:
			self.pub_vicon.publish(msg)
	def cbGoal(self,msg):
		if self.publish_goal_flag:
			self.pub_goal.publish(msg)

	def cbSrvPass(self,req):
		self.publish_vicon_flag = True
		self.publish_goal_flag = True
		rospy.loginfo("[%s][cbSrvPass] Passing all msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvCut(self,req):
		self.publish_vicon_flag = False
		self.publish_goal_flag = False
		rospy.loginfo("[%s][cbSrvCut] Cutting all msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvPassVicon(self,req):
		self.publish_vicon_flag = True
		rospy.loginfo("[%s] [cbSrvPassVicon] Passing vicon msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvCutVicon(self,req):
		self.publish_vicon_flag = False
		rospy.loginfo("[%s] [cbSrvCutVicon] Cutting vicon msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvPassGoal(self,req):
		self.publish_goal_flag = True
		rospy.loginfo("[%s] [cbSrvPassGoal] Passing goal msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvCutGoal(self,req):
		self.publish_goal_flag = False
		rospy.loginfo("[%s][cbSrvCutGoal] Cutting goal msgs." %(self.node_name))
		return EmptyResponse()

	def cbSrvCutTimed(self,req):
		rospy.loginfo("[%s][cbSrvCutTimed] Cutting all msgs." %(self.node_name))
		self.publish_vicon_flag = False
		self.publish_goal_flag = False
		t = 0.7
		rospy.sleep(t)
		rospy.loginfo("[%s][cbSrvCutTimed] Passing all msgs." %(self.node_name))
		self.publish_vicon_flag = True
		self.publish_goal_flag = True
		return EmptyResponse()

	def on_shutdown(self):
		rospy.loginfo("[%s] Shutting down." %(self.node_name))

if __name__ == '__main__':
    # Initialize the node with rospy
    rospy.init_node('vicon_switch_node', anonymous=False)
    # Create the NodeName object
    node = ViconSwitchNode()
    # Setup proper shutdown behavior 
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()