#!/usr/bin/env python

import rospy

if __name__ == '__main__':
    print("Init node")
    rospy.init_node('testNode', anonymous=True)
    print("Node initialized")