from dataclasses import dataclass

import numpy as np
import torch
from autoware_perception_msgs.msg import TrackedObjects, TrafficLightGroupArray
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation


@dataclass
class TrackingObject:
    kinematics_list: list
    shape_list: list
    class_label: int


def parse_timestamp(stamp) -> int:
    return stamp.sec * int(1e9) + stamp.nanosec


def get_nearest_msg(msg_list: list, stamp):
    """
    Get the nearest message and its index from a list of messages
    """
    stamp_int = parse_timestamp(stamp)
    nearest_msg = None
    nearest_index = -1
    nearest_time_diff = float("inf")
    for i, msg in enumerate(msg_list):
        msg_stamp = msg.header.stamp if hasattr(msg, "header") else msg.stamp
        msg_stamp_int = parse_timestamp(msg_stamp)
        time_diff = abs(msg_stamp_int - stamp_int)
        if time_diff < nearest_time_diff:
            nearest_time_diff = time_diff
            nearest_msg = msg
            nearest_index = i
    return nearest_msg, nearest_index


def get_transform_matrix(msg: Odometry):
    ego_x = msg.pose.pose.position.x
    ego_y = msg.pose.pose.position.y
    ego_z = msg.pose.pose.position.z
    ego_qx = msg.pose.pose.orientation.x
    ego_qy = msg.pose.pose.orientation.y
    ego_qz = msg.pose.pose.orientation.z
    ego_qw = msg.pose.pose.orientation.w
    rot = Rotation.from_quat([ego_qx, ego_qy, ego_qz, ego_qw])
    translation = np.array([ego_x, ego_y, ego_z])
    transform_matrix = rot.as_matrix()

    bl2map_matrix_4x4 = np.eye(4)
    bl2map_matrix_4x4[:3, :3] = transform_matrix
    bl2map_matrix_4x4[:3, 3] = translation

    map2bl_matrix_4x4 = np.eye(4)
    map2bl_matrix_4x4[:3, :3] = transform_matrix.T
    map2bl_matrix_4x4[:3, 3] = -transform_matrix.T @ translation
    return bl2map_matrix_4x4, map2bl_matrix_4x4


def pose_to_mat4x4(pose):
    """
    Convert ROS Pose to 4x4 matrix
    """
    mat = np.array(
        [
            [1.0, 0.0, 0.0, pose.position.x],
            [0.0, 1.0, 0.0, pose.position.y],
            [0.0, 0.0, 1.0, pose.position.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    rot = Rotation.from_quat(q)
    mat[:3, :3] = rot.as_matrix()
    return mat


def rot3x3_to_heading_cos_sin(rot3x3):
    """
    Convert 3x3 rotation matrix to heading cos and sin
    """
    rot = Rotation.from_matrix(rot3x3)
    heading = rot.as_euler("zyx")[0]
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    return cos_heading, sin_heading


def create_current_ego_state(kinematic_state_msg, acceleration_msg, wheel_base):
    ego_twist_linear = kinematic_state_msg.twist.twist.linear
    ego_twist_angular = kinematic_state_msg.twist.twist.angular
    ego_twist_linear = np.array([ego_twist_linear.x, ego_twist_linear.y, ego_twist_linear.z])
    ego_twist_angular = np.array([ego_twist_angular.x, ego_twist_angular.y, ego_twist_angular.z])
    linear_vel_norm = np.linalg.norm(ego_twist_linear)
    if abs(linear_vel_norm) < 0.2:
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
        steering_angle = 0.0
    else:
        yaw_rate = ego_twist_angular[2]
        steering_angle = np.arctan(yaw_rate * wheel_base / abs(linear_vel_norm))
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

    ego_current_state = torch.zeros((1, 10))
    ego_current_state[0, 0] = 0  # x in base_link is always 0
    ego_current_state[0, 1] = 0  # y in base_link is always 0
    ego_current_state[0, 2] = 1  # heading cos in base_link is always 1
    ego_current_state[0, 3] = 0  # heading sin in base_link is always 0
    ego_current_state[0, 4] = ego_twist_linear[0]  # velocity x
    ego_current_state[0, 5] = ego_twist_linear[1]  # velocity y
    ego_current_state[0, 6] = acceleration_msg.accel.accel.linear.x
    ego_current_state[0, 7] = acceleration_msg.accel.accel.linear.y
    ego_current_state[0, 8] = steering_angle  # steering angle
    ego_current_state[0, 9] = yaw_rate  # yaw rate
    return ego_current_state


def tracking_one_step(msg: TrackedObjects, tracked_objs: dict) -> dict:
    updated_tracked_objs = {}
    label_map = {
        0: 0,  # unknown -> vehicle
        1: 0,  # car -> vehicle
        2: 0,  # truck -> vehicle
        3: 0,  # bus -> vehicle
        4: 0,  # trailer -> vehicle
        5: 2,  # motorcycle -> bicycle
        6: 2,  # bicycle -> bicycle
        7: 1,  # pedestrian -> pedestrian
    }
    for i in range(len(msg.objects)):
        obj = msg.objects[i]
        object_id_bytes = bytes(obj.object_id.uuid)
        classification = obj.classification
        label_list = [i.label for i in classification]
        probability_list = [i.probability for i in classification]
        max_index = np.argmax(probability_list)
        label = label_list[max_index]
        label_in_model = label_map[label]
        kinematics = obj.kinematics
        shape = obj.shape
        if object_id_bytes in tracked_objs:
            tracked_obj = tracked_objs[object_id_bytes]
            tracked_obj.shape_list.append(shape)
            tracked_obj.kinematics_list.append(kinematics)
            updated_tracked_objs[object_id_bytes] = tracked_obj
        else:
            updated_tracked_objs[object_id_bytes] = TrackingObject(
                kinematics_list=[kinematics],
                shape_list=[shape],
                class_label=label_in_model,
            )

    return updated_tracked_objs


def convert_tracked_objects_to_tensor(
    tracked_objs: dict,
    map2bl_matrix_4x4: np.ndarray,
    max_num_objects: int,
    max_timesteps: int,
) -> torch.Tensor:
    neighbor = torch.zeros((1, max_num_objects, max_timesteps, 11))

    # Sort tracked objects by distance from ego
    # It is needed because the neighbors are sorted by distance from ego in the original code
    # https://github.com/SakodaShintaro/Diffusion-Planner/blob/6c8954e6424107dc1355ce7ba1d7aec10e6c8a1a/diffusion_planner/data_process/agent_process.py#L279-L280
    def sort_key(item):
        _, tracked_obj = item
        last_kinematics = tracked_obj.kinematics_list[-1]
        pose_in_map = pose_to_mat4x4(last_kinematics.pose_with_covariance.pose)
        pose_in_bl = map2bl_matrix_4x4 @ pose_in_map
        return np.linalg.norm(pose_in_bl[0:2, 3])

    tracked_objs = dict(sorted(tracked_objs.items(), key=sort_key))

    for i, (object_id_bytes, tracked_obj) in enumerate(tracked_objs.items()):
        if i >= max_num_objects:
            break
        label_in_model = tracked_obj.class_label
        for j in range(max_timesteps):
            if j < len(tracked_obj.kinematics_list):
                kinematics = tracked_obj.kinematics_list[-(j + 1)]
                shape = tracked_obj.shape_list[-(j + 1)]
            else:
                kinematics = tracked_obj.kinematics_list[0]
                shape = tracked_obj.shape_list[0]
            pose_in_map_4x4 = pose_to_mat4x4(kinematics.pose_with_covariance.pose)
            pose_in_bl_4x4 = map2bl_matrix_4x4 @ pose_in_map_4x4
            cos, sin = rot3x3_to_heading_cos_sin(pose_in_bl_4x4[0:3, 0:3])
            twist_in_map_4x4 = np.eye(4)
            twist_in_map_4x4[0, 3] = kinematics.twist_with_covariance.twist.linear.x
            twist_in_map_4x4[1, 3] = kinematics.twist_with_covariance.twist.linear.y
            twist_in_map_4x4[2, 3] = kinematics.twist_with_covariance.twist.linear.z
            twist_in_bl_4x4 = map2bl_matrix_4x4 @ twist_in_map_4x4
            neighbor[0, i, 20 - j, 0] = pose_in_bl_4x4[0, 3]  # x
            neighbor[0, i, 20 - j, 1] = pose_in_bl_4x4[1, 3]  # y
            neighbor[0, i, 20 - j, 2] = cos  # heading cos
            neighbor[0, i, 20 - j, 3] = sin  # heading sin
            neighbor[0, i, 20 - j, 4] = twist_in_bl_4x4[0, 3]  # velocity x
            neighbor[0, i, 20 - j, 5] = twist_in_bl_4x4[1, 3]  # velocity y
            # I don't know why but sometimes the length and width from autoware are 0
            neighbor[0, i, 20 - j, 6] = max(shape.dimensions.x, 1.0)  # length
            neighbor[0, i, 20 - j, 7] = max(shape.dimensions.y, 1.0)  # width
            neighbor[0, i, 20 - j, 8] = label_in_model == 0  # vehicle
            neighbor[0, i, 20 - j, 9] = label_in_model == 1  # pedestrian
            neighbor[0, i, 20 - j, 10] = label_in_model == 2  # bicycle
    return neighbor


def tracking(tracked_list: list[TrackedObjects]):
    tracked_objs = {}
    for i in range(len(tracked_list)):
        msg = tracked_list[i]
        updated_tracked_objs = tracking_one_step(msg, tracked_objs)
    return updated_tracked_objs


def convert_prediction_to_msg(pred: torch.Tensor, bl2map_matrix_4x4: np.array, stamp) -> Trajectory:
    # Convert to Trajectory message
    trajectory_msg = Trajectory()
    trajectory_msg.header.stamp = stamp
    trajectory_msg.header.frame_id = "map"
    trajectory_msg.points = []
    dt = 0.1
    prev_x = prev_y = 0
    for i in range(pred.shape[0]):
        point = TrajectoryPoint()

        # position
        curr_x = pred[i, 0]
        curr_y = pred[i, 1]
        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        vec3d = [curr_x, curr_y, 0.0]
        vec3d = bl2map_matrix_4x4 @ np.array([*vec3d, 1.0])
        point.pose.position.x = vec3d[0]
        point.pose.position.y = vec3d[1]
        point.pose.position.z = vec3d[2]

        # orientation
        curr_heading = pred[i, 2]
        rot = Rotation.from_euler("z", curr_heading, degrees=False).as_matrix()
        rot = bl2map_matrix_4x4[0:3, 0:3] @ rot
        quat = Rotation.from_matrix(rot).as_quat()
        point.pose.orientation.x = quat[0]
        point.pose.orientation.y = quat[1]
        point.pose.orientation.z = quat[2]
        point.pose.orientation.w = quat[3]

        # time/velocity
        seconds_float = float(i * dt)
        seconds_int = int(seconds_float)
        nanosec = int((seconds_float - seconds_int) * 1e9)
        point.time_from_start = Duration()
        point.time_from_start.sec = seconds_int
        point.time_from_start.nanosec = nanosec
        point.longitudinal_velocity_mps = distance / dt
        trajectory_msg.points.append(point)

        prev_x = curr_x
        prev_y = curr_y

    return trajectory_msg


def parse_traffic_light_recognition(msg: TrafficLightGroupArray):
    traffic_light_recognition = {}
    for traffic_light_group in msg.traffic_light_groups:
        traffic_light_group_id = traffic_light_group.traffic_light_group_id
        elements = traffic_light_group.elements
        if len(elements) == 1:
            traffic_light_recognition[traffic_light_group_id] = elements[0].color
        else:
            print("Traffic light group has more than one element.")
            # Ex.) RED(color=1) for CIRCLE(shape=1), but GREEN(color=3) for RIGHT_ARROW(shape=3)
            # In this case, we take the minimum value of the color
            traffic_light_recognition[traffic_light_group_id] = 4  # (WHITE)
            for element in elements:
                print(f"{element=}")
                traffic_light_recognition[traffic_light_group_id] = min(
                    traffic_light_recognition[traffic_light_group_id], element.color
                )
    return traffic_light_recognition
