ACL Joy Demo
============

This ROS package acts as a high-level motion planner and can be used with the [`snap-stack`](https://gitlab.com/mit-acl/fsw/snap-stack). It subscribes to a joystick (or keyboard) and generates the position / velocity [goal](https://gitlab.com/mit-acl/fsw/snap-stack/snapstack_msgs/blob/6e91e9af/msg/QuadGoal.msg) for the [`outer_loop`](https://gitlab.com/mit-acl/fsw/snap-stack/outer_loop).

## Getting Started

### Dependencies

You will need [`snapstack_msgs`](https://gitlab.com/mit-acl/fsw/snap-stack/snapstack_msgs) in your catkin workspace.

You will need to install pygame:

```bash
pip install -U pygame --user
```

#### Simulation Quick Setup

Using `wstool`, you may quickly clone all relevant dependencies into your workspace. If any of the dependent packages already exist, they will not be re-cloned. Install [`wstool`](http://wiki.ros.org/wstool#Installation) with `sudo apt install python-wstool`. Then,

```bash
$ cd ~/acl_ws/src   # or whatever workspace you are using
$ git clone git@gitlab.com:mit-acl/fsw/demos/acl_joy.git
$ wstool init
$ wstool merge acl_joy/.snapsim.rosinstall
$ wstool update -j8 # can be repeated to keep tracked repos up-to-date
```

### Running

If you have an Xbox controller connected, you may run

```bash
roslaunch acl_joy joy.launch
```

Press A to take off, X to land, and B to kill the motors. If you do not have a joystick and would like to use your keyboard (only recommended for simulation), run

```bash
roslaunch acl_joy key.launch
```

Press 1 to take off, 2 to land, and 3 to kill the motors.

**Note**: In either case make sure you have the appropriate `veh:=` and `num:=` flags set.

## FAQ

1. My joystick is on and is paired with the received (hold down the appropriate buttons to pair; a paired xbox controller has a solid light). However, I do not see any values on `rostopic echo /<veh><num>/joy`.

    - It could be your device. You can look at `ls /dev/js*` to see which joystick devices your computer sees. The default device used is `/dev/js0`. Change with `dev:=js1` launch flag.
