Snap Autopilot
==============

`snap` is a customizable flight stack for multirotors equipped with one of Qualcomm's Snapdragon Flight board ([original](https://docs.px4.io/v1.9.0/en/flight_controller/snapdragon_flight.html) or [Pro](https://developer.qualcomm.com/hardware/qualcomm-flight-pro)). The position, velocity, attitude, and IMU biases are estimated by fusing propogated IMU measurments with an external measurement (e.g., mocap, VIO, etc.) via an observer. Attitude control is achieved with a quaternion-based controller. `snap` uses the [`esc_interface`](https://gitlab.com/mit-acl/fsw/snap-stack/esc_interface) shared library that enables the Snapdragon to interface with many commerically available ESCs via PWM.

## Getting Started

The Snapdragon Flight board contains a CPU and a DSP. The DSP is used to interface with hardware IO while the CPU is used for ROS and other standard embedded Linux uses. The following quickly suggests how to get started with development.

### Development Environment

Setup your development environment following instructions in either [`sf-dev`](https://gitlab.com/mit-acl/fsw/snap-stack/sf-dev) or [`sfpro-dev`](https://gitlab.com/mit-acl/fsw/snap-stack/sfpro-dev) depending on your version of the Snapdragon board.

For the new snapdragon, you must use the `sfpro-dev` to compile CPU and DSP code.

For the old snapdragon, `sf-dev` is only necessary for compiling DSP code. CPU code can be built as normal onboard the snapdragon via SSH/adb. Create a Catkin workspace (prefer `catkin build` over `catkin_make`) to clone this project and its dependencies into.

### Dependencies

Standard [`snap-stack`](https://gitlab.com/mit-acl/fsw/snap-stack) are required. Your catkin_ws should have at least the following

```bash
simulation_ws
└── src
    ├── comm_monitor
    ├── outer_loop
    ├── snap
    └── snapstack_msgs
```

In addition, ensure that `libncurses5-dev` is installed (`sudo apt install`).

Make sure to install the dependencies' dependencies as well!

### Building

Build with `catkin build`. Source your workspace with `source devel/setup.bash`.

### Running

In separate snapdragon terminals:

1. Start the IMU driver

    ```bash
    imu_app -s 2   # 500 Hz
    ```

2. Start the autopilot

    ```bash
    roslaunch snap snap.launch
    ```

    Note that the `veh:=`, `num:=`, and `sfpro:=` args are automagically determined by environment variables.

3. Start the `outer_loop`:

    ```bash
    roslaunch outer_loop cntrl.launch
    ```

4. For safety, you must arm the motors using the `/<veh><num>/arm` service call. For convenience, you can use the `esc_interface` node to do this with the spacebar. Run:

    ```bash
    roslaunch snap esc.launch
    ```

## Inputs/Outputs

### Subscribed Topics
 * `<vehicle name>/pose`: Position and orientation update for estimator
 * `<vehicle name>/attCmds`: Desired attitude and angular rates

### Published Topics
 * `<vehicle name>/imu`: IMU data
 * `<vehicle name>/state`: Vehicle's full state
 * `<vehicle name>/motors`: Motor commands
 * `<vehicle name>/smc`: Attitude controller data
 
**NOTE**: The snap autopilot expects an *ENU* fixed frame and uses body *flu* coordinate system. Ensure that the input pose is transformed in the correct frame.
