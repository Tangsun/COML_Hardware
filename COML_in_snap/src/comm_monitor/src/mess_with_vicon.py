#!/usr/bin/env python
import rospy
from copy import deepcopy
from geometry_msgs.msg import PoseStamped
from collections import deque

class MessWithViconNode():
	def __init__(self):
		self.node_name = rospy.get_name()
		self.pub_pose = rospy.Publisher("~outgoing_pose",PoseStamped,queue_size=1)
		self.sub_pose = rospy.Subscriber("~incoming_pose",PoseStamped,self.cbNewViconMsg)

		# Get all params from launch file
		self.mode = rospy.get_param("~mode","publish_1_of_x")
		self.x = rospy.get_param("~x",1)
		self.y = rospy.get_param("~y",1)

		self.count = deepcopy(self.x)
		self.poses = deque()

	def cbNewViconMsg(self,vicon_msg):
		if self.mode == "publish_1_of_x":
			self.count -= 1
			if self.count == 0:
				self.count = deepcopy(self.x)
				self.pub_pose.publish(vicon_msg)
		elif self.mode == "delay_for_y":
			self.poses.append(vicon_msg)
			rospy.Timer(rospy.Duration(self.y),self.pop_and_publish,oneshot=True)

	def pop_and_publish(self,event):
		self.pub_pose.publish(self.poses.popleft())


	def on_shutdown(self):
		rospy.loginfo("[%s] Shutting down." %(self.node_name))

if __name__ == '__main__':
    # Initialize the node with rospy
    rospy.init_node('mess_with_vicon_node', anonymous=False)

    # Create the NodeName object
    node = MessWithViconNode()

    # Setup proper shutdown behavior 
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()