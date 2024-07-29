#!/bin/bash

# Activate form COML_in_snap directory
# source src/outer_loop_python/src/batch_snap_sim_test.sh

export MODEL_DIRECTORY="./src/outer_loop_python/models/"
export SHELL_PID=$$

catkin build
source ./devel/setup.bash

# Walk through the MODEL_DIRECTORY
find "$MODEL_DIRECTORY" -type f -name "*.pkl" | while read file
do
    for wind_speed in 6 9 12
    do
        export MAX_WIND=$wind_speed
        for traj_type in 'spline'
        do
            echo "Testing Model $file"
            export TRAJ_TYPE=$traj_type

            export PKL_FILENAME="$(basename "$file")"
            export PKL_TRIAL_NAME="$(basename "$(dirname "$file")")"
            roslaunch snap_sim sim.launch rviz:=false behavior_gui:=false
            # How to automatically exit
        done
    done

    export MAX_WIND=$wind_speed
    for traj_type in 'circle' 'figure_eight'
    do
        echo "Testing Model $file"
        export TRAJ_TYPE=$traj_type

        export PKL_FILENAME="$(basename "$file")"
        export PKL_TRIAL_NAME="$(basename "$(dirname "$file")")"
        roslaunch snap_sim sim.launch rviz:=false behavior_gui:=false
        # How to automatically exit
    done
done

unset PKL_FILENAME
unset PKL_TRIAL_NAME
unset TRAJ_TYPE
unset MAX_WIND