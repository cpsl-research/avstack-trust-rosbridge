from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    trust_measurement_config = PathJoinSubstitution(
        [
            get_package_share_directory("avtrust_bridge"),
            "config",
            "trust_measurement.yaml",
        ]
    )

    trust_updater_config = PathJoinSubstitution(
        [
            get_package_share_directory("avtrust_bridge"),
            "config",
            "trust_updater.yaml",
        ]
    )

    trust_measurement = Node(
        package="avtrust_bridge",
        executable="measurement",
        name="trust_estimator",
        parameters=[trust_measurement_config],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    trust_updater = Node(
        package="avtrust_bridge",
        executable="updater",
        name="trust_updater",
        parameters=[trust_updater_config],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    return LaunchDescription([trust_measurement, trust_updater])
