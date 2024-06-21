Comm Monitor
============

This ROS package logs the communication latency from base station. It is meant to be run on-board the companion computer of a vehicle. 

### Method

Each time a message (from mocap or base station goal) is received, a timestamp is updated with the current clock of the companion computer (i.e., the receive time). At a fixed rate (`comm_monitor_hz`, default 100 Hz), the difference between the current companion computer clock and the last receive time is published as `ages`.
