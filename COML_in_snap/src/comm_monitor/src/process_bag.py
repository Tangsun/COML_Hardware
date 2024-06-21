#!/usr/bin/env python
import rosbag
import matplotlib.pyplot as plt
bag = rosbag.Bag('/home/mfe/Downloads/vicon_drop_out_nom_2.bag')

i=0
lag = []
tlist=[]
for topic, msg, t in bag.read_messages(topics=['/RQ06/pose']):
	lag.append(msg.header.stamp.to_sec())
	tlist.append(t.to_sec())
bag.close()

plt.plot(tlist,lag)
plt.xlabel('Received Time (odroid)')
plt.ylabel('Vicon Time')
plt.show()