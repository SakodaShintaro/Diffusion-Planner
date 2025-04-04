#!/bin/bash

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 run diffusion_planner_ros diffusion_planner_node --ros-args \
    -p vector_map_path:=/home/shintarosakoda/data/misc/20250329_psim_rosbag/map/lanelet2_map.osm \
    -p config_json_path:=/media/shintarosakoda/5EA85517A854EF51/diffusion_planner_training_result/train_result/2025-03-20-180651_datasize_1M/args.json \
    -p ckpt_path:=/media/shintarosakoda/5EA85517A854EF51/diffusion_planner_training_result/train_result/2025-03-20-180651_datasize_1M/latest.pth
