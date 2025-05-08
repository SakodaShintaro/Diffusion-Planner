"""Make markers to visualize in rviz."""

import numpy as np
import torch
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def create_trajectory_marker(trajectory_msg):
    marker_array = MarkerArray()

    # Points marker
    points_marker = Marker()
    points_marker.header = trajectory_msg.header
    points_marker.ns = "trajectory_points"
    points_marker.id = 1
    points_marker.type = Marker.SPHERE_LIST
    points_marker.action = Marker.ADD
    points_marker.pose.orientation.w = 1.0
    points_marker.scale.x = 0.4
    points_marker.scale.y = 0.4
    points_marker.scale.z = 0.4
    points_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # Red
    points_marker.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1 sec

    # Arrows marker
    arrows_marker = Marker()
    arrows_marker.header = trajectory_msg.header
    arrows_marker.ns = "trajectory_arrows"
    arrows_marker.id = 2
    arrows_marker.type = Marker.ARROW
    arrows_marker.action = Marker.ADD
    arrows_marker.scale.x = 0.3
    arrows_marker.scale.y = 0.5
    arrows_marker.scale.z = 0.5
    arrows_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)  # Blue
    arrows_marker.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1 sec
    arrows_marker.pose.orientation.w = 1.0

    # Markers for 1-second intervals
    time_markers = Marker()
    time_markers.header = trajectory_msg.header
    time_markers.ns = "trajectory_time_markers"
    time_markers.id = 3
    time_markers.type = Marker.SPHERE_LIST
    time_markers.action = Marker.ADD
    time_markers.pose.orientation.w = 1.0
    time_markers.scale.x = 0.6
    time_markers.scale.y = 0.6
    time_markers.scale.z = 0.6
    time_markers.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # Yellow
    time_markers.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1 sec

    for i, point in enumerate(trajectory_msg.points):
        p = Point()
        p.x = point.pose.position.x
        p.y = point.pose.position.y
        p.z = point.pose.position.z

        points_marker.points.append(p)

        if (i + 1) % 10 == 0:
            time_markers.points.append(p)

        # Arrow marker (direction)
        if i % 20 == 0:
            start_point = Point()
            start_point.x = point.pose.position.x
            start_point.y = point.pose.position.y
            start_point.z = point.pose.position.z
            arrows_marker.points.append(start_point)

            q = [
                point.pose.orientation.x,
                point.pose.orientation.y,
                point.pose.orientation.z,
                point.pose.orientation.w,
            ]
            rot = Rotation.from_quat(q)
            direction = rot.as_matrix()[:, 0]

            end_point = Point()
            end_point.x = start_point.x + direction[0] * 1.0
            end_point.y = start_point.y + direction[1] * 1.0
            end_point.z = start_point.z + direction[2] * 1.0
            arrows_marker.points.append(end_point)

    marker_array.markers.append(points_marker)
    marker_array.markers.append(arrows_marker)
    marker_array.markers.append(time_markers)

    return marker_array


def create_route_marker(route_tensor: torch.Tensor, stamp) -> MarkerArray:
    marker_array = MarkerArray()
    for j in range(route_tensor.shape[1]):
        centerline_marker = Marker()
        centerline_marker.header.stamp = stamp
        centerline_marker.header.frame_id = "base_link"
        centerline_marker.ns = "route"
        centerline_marker.id = j
        centerline_marker.type = Marker.LINE_STRIP
        centerline_marker.action = Marker.ADD
        centerline_marker.pose.orientation.w = 1.0
        centerline_marker.scale.x = 0.6
        centerline_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        centerline_marker.lifetime = Duration(sec=1, nanosec=0)
        centerline_marker.points = []
        centerline_in_base_link = route_tensor[0, j, :, :2].cpu().numpy()
        if np.sum(centerline_in_base_link) == 0:
            continue
        centerline_in_base_link = np.concatenate(
            [
                centerline_in_base_link,
                np.zeros((centerline_in_base_link.shape[0], 1)),
                np.ones((centerline_in_base_link.shape[0], 1)),
            ],
            axis=1,
        )
        # Create a marker for the centerline
        for i, point in enumerate(centerline_in_base_link):
            p = Point()
            norm = np.linalg.norm(point)
            if norm < 2:
                continue
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            centerline_marker.points.append(p)
        marker_array.markers.append(centerline_marker)
    return marker_array


def create_neighbor_marker(neighbor_tensor: torch.Tensor, stamp) -> MarkerArray:
    """
    Create a marker array to visualize neighboring objects (vehicles, pedestrians, bicycles).

    Args:
        neighbor_tensor: (batch(1), num_objects(32), num_frames(21), features(11))
                        Last Dim: [x, y, cos, sin, vx, vy, length, width, vehicle, pedestrian, bicycle]
        stamp:
    """
    marker_array = MarkerArray()

    # Last frame data
    neighbor_data = neighbor_tensor[0, :, -1, :]  # (32, 11)

    # counter
    marker_id = 0

    for i in range(neighbor_data.shape[0]):
        # if the object has 0 values, skip it
        if torch.sum(torch.abs(neighbor_data[i, :2])) < 1e-6:
            continue

        x = neighbor_data[i, 0].item()
        y = neighbor_data[i, 1].item()
        cos_h = neighbor_data[i, 2].item()
        sin_h = neighbor_data[i, 3].item()
        heading = np.arctan2(sin_h, cos_h)
        width = neighbor_data[i, 6].item()
        length = neighbor_data[i, 7].item()
        obj_type_idx = torch.argmax(neighbor_data[i, 8:11]).item()

        colors = [
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7),  # Vehicle - Blue
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7),  # Pedestrian - Green
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.7),  # Bicycle - Magenta
        ]

        # Cube marker
        cube_marker = Marker()
        cube_marker.header.stamp = stamp
        cube_marker.header.frame_id = "base_link"
        cube_marker.ns = "neighbor_objects"
        cube_marker.id = marker_id
        marker_id += 1
        cube_marker.type = Marker.CUBE
        cube_marker.action = Marker.ADD

        cube_marker.pose.position.x = x
        cube_marker.pose.position.y = y
        cube_marker.pose.position.z = 0.5

        q = Rotation.from_euler("z", heading).as_quat()
        cube_marker.pose.orientation.x = q[0]
        cube_marker.pose.orientation.y = q[1]
        cube_marker.pose.orientation.z = q[2]
        cube_marker.pose.orientation.w = q[3]

        cube_marker.scale.x = length
        cube_marker.scale.y = width
        cube_marker.scale.z = 1.5

        cube_marker.color = colors[obj_type_idx]

        cube_marker.lifetime = Duration(sec=0, nanosec=300000000)

        marker_array.markers.append(cube_marker)

        # Text marker
        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = "base_link"
        text_marker.ns = "neighbor_labels"
        text_marker.id = marker_id
        marker_id += 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = 2.0
        obj_types = ["Vehicle", "Pedestrian", "Bicycle"]
        text_marker.text = f"{obj_types[obj_type_idx]} #{i}"
        text_marker.scale.z = 0.8
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        text_marker.lifetime = Duration(sec=0, nanosec=300000000)
        marker_array.markers.append(text_marker)

        # History marker
        path_marker = Marker()
        path_marker.header.stamp = stamp
        path_marker.header.frame_id = "base_link"
        path_marker.ns = "neighbor_paths"
        path_marker.id = marker_id
        marker_id += 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        # for 10 frames
        history_length = min(10, neighbor_tensor.shape[2] - 1)
        for j in range(history_length):
            frame_idx = -j - 1
            past_data = neighbor_tensor[0, i, frame_idx, :]

            # if the object has 0 values, skip it
            if torch.sum(torch.abs(past_data[:2])) > 1e-6:
                p = Point()
                p.x = past_data[0].item()
                p.y = past_data[1].item()
                p.z = 0.1
                path_marker.points.append(p)

        if len(path_marker.points) > 1:
            path_marker.scale.x = 0.1
            path_marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.5)
            path_marker.lifetime = Duration(sec=0, nanosec=300000000)
            marker_array.markers.append(path_marker)

    return marker_array
