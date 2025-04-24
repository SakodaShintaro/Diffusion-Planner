import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rosbag2_py
from autoware_perception_msgs.msg import (
    TrackedObjects,
    TrafficLightGroupArray,
)
from autoware_planning_msgs.msg import LaneletRoute
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    create_lane_tensor,
)
from diffusion_planner_ros.utils import (
    convert_tracked_objects_to_tensor,
    create_current_ego_state,
    get_nearest_msg,
    get_transform_matrix,
    parse_timestamp,
    parse_traffic_light_recognition,
    tracking_one_step,
)
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tqdm import tqdm


@dataclass
class FrameData:
    timestamp: int
    tracked_objects: TrackedObjects
    kinematic_state: Odometry
    acceleration: AccelWithCovarianceStamped
    traffic_signals: TrafficLightGroupArray


@dataclass
class SequenceData:
    """This class means one sequence of data.
    It contains exactly one route msg and multiple other msgs.
    A rosbag may contain multiple sequences.
    """

    data_list: list[FrameData]
    route: LaneletRoute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def tracking_list(tracking_object_msg_list: list):
    tracking_obj = {}
    for tracking_object_msg in tracking_object_msg_list:
        tracking_obj = tracking_one_step(tracking_object_msg, tracking_obj)
    return tracking_obj


def create_ego_future(data_list, i, future_time_steps, map2bl_matrix_4x4):
    ego_future_x = []
    ego_future_y = []
    ego_future_yaw = []
    for j in range(future_time_steps):
        x = data_list[i + j + 1].kinematic_state.pose.pose.position.x
        y = data_list[i + j + 1].kinematic_state.pose.pose.position.y
        z = data_list[i + j + 1].kinematic_state.pose.pose.position.z
        qx = data_list[i + j + 1].kinematic_state.pose.pose.orientation.x
        qy = data_list[i + j + 1].kinematic_state.pose.pose.orientation.y
        qz = data_list[i + j + 1].kinematic_state.pose.pose.orientation.z
        qw = data_list[i + j + 1].kinematic_state.pose.pose.orientation.w
        rot = Rotation.from_quat([qx, qy, qz, qw])
        pose_in_map = np.eye(4)
        pose_in_map[:3, :3] = rot.as_matrix()
        pose_in_map[0, 3] = x
        pose_in_map[1, 3] = y
        pose_in_map[2, 3] = z
        pose_in_bl = map2bl_matrix_4x4 @ pose_in_map
        ego_future_x.append(pose_in_bl[0, 3])
        ego_future_y.append(pose_in_bl[1, 3])
        rot = Rotation.from_matrix(pose_in_bl[:3, :3])
        yaw = rot.as_euler("xyz")[2]
        ego_future_yaw.append(yaw)

    ego_future_np = np.concatenate(
        [
            np.array(ego_future_x).reshape(-1, 1),
            np.array(ego_future_y).reshape(-1, 1),
            np.array(ego_future_yaw).reshape(-1, 1),
        ],
        axis=1,
    )
    return ego_future_np


if __name__ == "__main__":
    args = parse_args()
    rosbag_path = args.rosbag_path
    vector_map_path = args.vector_map_path
    save_dir = args.save_dir
    step = args.step
    limit = args.limit

    vector_map = convert_lanelet(str(vector_map_path))

    # parse rosbag
    serialization_format = "cdr"
    storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    target_topic_list = [
        "/localization/kinematic_state",
        "/localization/acceleration",
        "/perception/object_recognition/tracking/objects",
        "/perception/traffic_light_recognition/traffic_signals",
        "/planning/mission_planning/route",
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

    route_msgs = topic_name_to_data["/planning/mission_planning/route"]
    sequence_data_list = [SequenceData([], route_msg) for route_msg in route_msgs]

    # convert to FrameData
    # The base topic is "/perception/object_recognition/tracking/objects" (10Hz)
    n = len(topic_name_to_data["/perception/object_recognition/tracking/objects"])
    print(f"{n=}")
    progress_bar = tqdm(total=n)
    for i in range(n):
        tracking = topic_name_to_data["/perception/object_recognition/tracking/objects"][i]
        timestamp = parse_timestamp(tracking.header.stamp)
        latest_msgs = {
            "/localization/kinematic_state": None,
            "/localization/acceleration": None,
            "/perception/traffic_light_recognition/traffic_signals": None,
        }

        ok = True
        for key in latest_msgs.keys():
            curr_msg, curr_index = get_nearest_msg(topic_name_to_data[key], tracking.header.stamp)
            if curr_msg is None:
                print(f"Cannot find {key} msg")
                ok = False
                break
            topic_name_to_data[key] = topic_name_to_data[key][curr_index:]
            latest_msgs[key] = curr_msg
            msg_stamp = curr_msg.header.stamp if hasattr(curr_msg, "header") else curr_msg.stamp
            msg_stamp_int = parse_timestamp(msg_stamp)
            diff = abs(timestamp - msg_stamp_int)
            if diff > int(0.1 * 1e9):
                print(f"Over 100 msec: {key} {len(topic_name_to_data[key])=}, {diff=:,}")
                ok = False

        # check kinematic_state
        kinematic_state = latest_msgs["/localization/kinematic_state"]
        covariance = kinematic_state.pose.covariance
        covariance_xx = covariance[0]
        covariance_yy = covariance[7]
        if covariance_xx > 1e-1 or covariance_yy > 1e-1:
            print(f"Invalid kinematic_state {covariance_xx=:.5f}, {covariance_yy=:.5f}")
            ok = False

        # check route
        # find the latest route msg
        max_route_index = -1
        max_route_timestamp = 0
        for j in range(len(route_msgs)):
            route_msg = route_msgs[j]
            route_stamp = parse_timestamp(route_msg.header.stamp)
            if route_stamp <= timestamp and route_stamp > max_route_timestamp:
                max_route_timestamp = route_stamp
                max_route_index = j
        if max_route_index == -1:
            print("Cannot find route msg")
            continue

        sequence = sequence_data_list[max_route_index]

        if not ok:
            if len(sequence.data_list) == 0:
                # At the beginning of recording, some msgs may be missing
                # Skip this frame
                print(f"Skip this frame {i=}/{n=}")
                continue
            else:
                # If the msg is missing in the middle of recording, we can use the msgs to this point
                print(f"Finish at this frame {i=}/{n=}")
                break

        sequence.data_list.append(
            FrameData(
                timestamp=timestamp,
                tracked_objects=tracking,
                kinematic_state=latest_msgs["/localization/kinematic_state"],
                acceleration=latest_msgs["/localization/acceleration"],
                traffic_signals=latest_msgs[
                    "/perception/traffic_light_recognition/traffic_signals"
                ],
            )
        )
        progress_bar.update(1)

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

    map_name = rosbag_path.stem
    sequence_num = len(sequence_data_list)
    print(f"Total {sequence_num} sequences")

    for seq_id, seq in enumerate(sequence_data_list):
        print(f"Processing sequence {seq_id + 1}/{sequence_num}")

        data_list = seq.data_list
        n = len(data_list)
        print(f"Total {n} frames")

        # if less than 1800 frames (= 3 min), skip this sequence
        if n < 1800:
            print(f"Skip this sequence because the number of frames {n} is less than 1800 frames")
            continue

        # list[FrameData] -> npz
        progress = tqdm(total=(n - PAST_TIME_STEPS - FUTURE_TIME_STEPS) // step)
        for i in range(PAST_TIME_STEPS, n - FUTURE_TIME_STEPS, step):
            token = f"{seq_id:08d}{i:08d}"

            # 2 sec tracking (for input)
            tracking_past = tracking_list(
                [frame_data.tracked_objects for frame_data in data_list[i - PAST_TIME_STEPS : i]]
            )
            # 8 sec tracking (for ground truth)
            tracking_future = tracking_list(
                [frame_data.tracked_objects for frame_data in data_list[i : i + FUTURE_TIME_STEPS]]
            )

            # filter tracking_future by indices in tracking_past
            tracking_past_keys = set(tracking_past.keys())
            filtered_tracking_future = {}
            for key in tracking_future.keys():
                if key in tracking_past_keys:
                    filtered_tracking_future[key] = tracking_future[key]

            bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(
                data_list[i].kinematic_state
            )

            traffic_light_recognition = parse_traffic_light_recognition(
                data_list[i].traffic_signals
            )

            ego_tensor = create_current_ego_state(
                data_list[i].kinematic_state, data_list[i].acceleration, wheel_base=5.0
            ).squeeze(0)

            ego_future_np = create_ego_future(data_list, i, FUTURE_TIME_STEPS, map2bl_matrix_4x4)

            neighbor_past_tensor = convert_tracked_objects_to_tensor(
                tracked_objs=tracking_past,
                map2bl_matrix_4x4=map2bl_matrix_4x4,
                max_num_objects=NEIGHBOR_NUM,
                max_timesteps=PAST_TIME_STEPS,
            ).squeeze(0)

            neighbor_future_tensor = convert_tracked_objects_to_tensor(
                tracked_objs=filtered_tracking_future,
                map2bl_matrix_4x4=map2bl_matrix_4x4,
                max_num_objects=NEIGHBOR_NUM,
                max_timesteps=FUTURE_TIME_STEPS,
            ).squeeze(0)
            # (32, 80, 11) -> (32, 80, 3)
            neighbor_future_tensor = neighbor_future_tensor[:, :, :4]
            # fixed cos(2) sin(3) -> heading
            neighbor_future_tensor[:, :, 2] = np.arctan2(
                neighbor_future_tensor[:, :, 3], neighbor_future_tensor[:, :, 2]
            )
            neighbor_future_tensor = neighbor_future_tensor[:, :, :3]

            static_objects = np.zeros((STATIC_NUM, 10), dtype=np.float32)

            lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
                vector_map.lane_segments.values(),
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                mask_range=100,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=70,
                dev="cpu",
            )

            target_segments = [
                vector_map.lane_segments[segment.preferred_primitive.id]
                for segment in seq.route.segments
            ]
            route_tensor, route_speed_limit, route_has_speed_limit = create_lane_tensor(
                target_segments,
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                mask_range=100,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=25,
                dev="cpu",
            )

            curr_data = {
                "map_name": map_name,
                "token": token,
                "ego_current_state": ego_tensor.numpy(),
                "ego_agent_future": ego_future_np,
                "neighbor_agents_past": neighbor_past_tensor.numpy(),
                "neighbor_agents_future": neighbor_future_tensor.numpy(),
                "static_objects": static_objects,
                "lanes": lanes_tensor.squeeze(0).numpy(),
                "lanes_speed_limit": lanes_speed_limit.squeeze(0).numpy(),
                "lanes_has_speed_limit": lanes_has_speed_limit.squeeze(0).numpy(),
                "route_lanes": route_tensor.squeeze(0).numpy(),
                "route_lanes_speed_limit": route_speed_limit.squeeze(0).numpy(),
                "route_lanes_has_speed_limit": route_has_speed_limit.squeeze(0).numpy(),
            }
            # save the data
            save_dir.mkdir(parents=True, exist_ok=True)
            output_file = f"{save_dir}/{map_name}_{token}.npz"
            np.savez(output_file, **curr_data)
            progress.update(1)
