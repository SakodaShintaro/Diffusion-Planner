import argparse
from pathlib import Path
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from collections import defaultdict
import numpy as np
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
)
from diffusion_planner_ros.utils import (
    create_current_ego_state,
    get_nearest_msg,
    parse_timestamp,
    tracking_one_step,
    convert_tracked_objects_to_tensor,
    get_transform_matrix,
)
import secrets
from dataclasses import dataclass
from autoware_perception_msgs.msg import (
    TrackedObjects,
    TrafficLightGroupArray,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelWithCovarianceStamped
from sensor_msgs.msg import CompressedImage


@dataclass
class FrameData:
    timestamp: int
    tracked_objects: TrackedObjects
    kinematic_state: Odometry
    acceleration: AccelWithCovarianceStamped
    traffic_signals: TrafficLightGroupArray
    image: CompressedImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def tracking_list(tracking_object_msg_list: list):
    tracking_obj = {}
    for tracking_object_msg in tracking_object_msg_list:
        tracking_obj = tracking_one_step(tracking_object_msg, tracking_obj)
    return tracking_obj


if __name__ == "__main__":
    args = parse_args()
    rosbag_path = args.rosbag_path
    vector_map_path = args.vector_map_path
    limit = args.limit

    vector_map = convert_lanelet(str(vector_map_path))

    serialization_format = "cdr"
    storage_options = rosbag2_py.StorageOptions(
        uri=str(rosbag_path), storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    target_topic_list = [
        "/localization/kinematic_state",
        "/localization/acceleration",
        "/perception/object_recognition/tracking/objects",
        "/perception/traffic_light_recognition/traffic_signals",
        "/planning/mission_planning/route",
        "/sensing/camera/camera0/image_rect_color/compressed",
    ]

    storage_filter = rosbag2_py.StorageFilter(topics=target_topic_list)
    reader.set_filter(storage_filter)

    topic_name_to_data = defaultdict(list)
    parse_num = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic in target_topic_list:
            topic_name_to_data[topic].append(msg)
            parse_num += 1
            if limit > 0 and parse_num >= limit:
                break

    for key, value in topic_name_to_data.items():
        print(f"{key}: {len(value)} msgs")

    # 最初にmsgsの10Hzでの整形(tracked_objects基準)を行う
    n = len(topic_name_to_data["/perception/object_recognition/tracking/objects"])
    data_list = []
    for i in range(n):
        tracking = topic_name_to_data[
            "/perception/object_recognition/tracking/objects"
        ][i]
        timestamp = parse_timestamp(tracking.header.stamp)
        latest_msgs = {
            "/localization/kinematic_state": None,
            "/localization/acceleration": None,
            "/perception/traffic_light_recognition/traffic_signals": None,
            "/sensing/camera/camera0/image_rect_color/compressed": None,
        }

        for key in latest_msgs.keys():
            curr_msg, curr_index = get_nearest_msg(
                topic_name_to_data[key], tracking.header.stamp
            )
            topic_name_to_data[key] = topic_name_to_data[key][curr_index:]
            latest_msgs[key] = curr_msg

        data_list.append(
            FrameData(
                timestamp=timestamp,
                tracked_objects=tracking,
                kinematic_state=latest_msgs["/localization/kinematic_state"],
                acceleration=latest_msgs["/localization/acceleration"],
                traffic_signals=latest_msgs[
                    "/perception/traffic_light_recognition/traffic_signals"
                ],
                image=latest_msgs[
                    "/sensing/camera/camera0/image_rect_color/compressed"
                ],
            )
        )

    """
    作りたいnpz
    map_name                    <U26    ()
    token                       <U16    ()
    ego_current_state           float32 (10,)
    ego_agent_future            float32 (80, 3)
    neighbor_agents_past        float32 (32, 21, 11)
    neighbor_agents_future      float32 (32, 80, 3)
    static_objects              float32 (5, 10)
    lanes                       float32 (70, 20, 12)
    lanes_speed_limit           float32 (70, 1)
    lanes_has_speed_limit       bool    (70, 1)
    route_lanes                 float32 (25, 20, 12)
    route_lanes_speed_limit     float32 (25, 1)
    route_lanes_has_speed_limit bool    (25, 1)
    """
    PAST_TIME_STEPS = 21
    FUTURE_TIME_STEPS = 80
    NEIGHBOR_NUM = 32
    STATIC_NUM = 5
    LANE_NUM = 70
    LANE_LEN = 20
    ROUTE_NUM = 25
    ROUTE_LEN = 20

    map_name = "autoware_map"

    # これをrosbagのデータから作る
    # 時刻の基準とするデータは "/perception/object_recognition/tracking/objects" (10Hz)
    # 重複が出ないように8秒ごとに作る
    for i in range(PAST_TIME_STEPS, n, FUTURE_TIME_STEPS):
        print(f"{i=}")
        # 2秒前からここまでのトラッキング（入力用）
        tracking_past = tracking_list(
            topic_name_to_data["/perception/object_recognition/tracking/objects"][
                i - PAST_TIME_STEPS : i
            ]
        )
        # 2秒前から8秒後までのトラッキング（GT用）
        tracking_future = tracking_list(
            topic_name_to_data["/perception/object_recognition/tracking/objects"][
                i - PAST_TIME_STEPS : i + FUTURE_TIME_STEPS
            ]
        )

        # filter tracking_future by indices in tracking_past
        tracking_past_keys = set(tracking_past.keys())
        filtered_tracking_future = {}
        for key in tracking_future.keys():
            if key in tracking_past_keys:
                filtered_tracking_future[key] = tracking_future[key]

        print(tracking_past.keys())
        print(filtered_tracking_future.keys())

        bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(
            data_list[i].kinematic_state
        )

        ego_tensor = create_current_ego_state(
            data_list[i].kinematic_state, data_list[i].acceleration, wheel_base=5.0
        )

        neighbor_past_tensor = convert_tracked_objects_to_tensor(
            tracked_objs=tracking_past,
            map2bl_matrix_4x4=map2bl_matrix_4x4,
            max_num_objects=NEIGHBOR_NUM,
            max_timesteps=PAST_TIME_STEPS,
        )

        neighbor_future_tensor = convert_tracked_objects_to_tensor(
            tracked_objs=filtered_tracking_future,
            map2bl_matrix_4x4=map2bl_matrix_4x4,
            max_num_objects=NEIGHBOR_NUM,
            max_timesteps=FUTURE_TIME_STEPS,
        )
