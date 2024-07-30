import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    trust_config = os.path.join(
        get_package_share_directory("trust"),
        "config",
        "mate.yaml",
    )

    trust_node = Node(
        package="trust",
        executable="mate",
        name="trust_estimator",
        parameters=[trust_config],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    return LaunchDescription([trust_node])
