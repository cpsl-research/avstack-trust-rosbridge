import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    trust_viz_config = os.path.join(
        get_package_share_directory("avtrust_bridge"),
        "config",
        "visualizer.yaml",
    )

    trust_viz_node = Node(
        package="avtrust_bridge",
        executable="visualizer",
        name="trust_visualizer",
        parameters=[trust_viz_config],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    agent_pub = Node(
        package="avtrust_bridge",
        executable="agent_pub_sample",
        name="agent_trust_sample",
    )

    track_pub = Node(
        package="avtrust_bridge",
        executable="track_pub_sample",
        name="track_trust_sample",
    )

    return LaunchDescription([trust_viz_node, agent_pub, track_pub])
