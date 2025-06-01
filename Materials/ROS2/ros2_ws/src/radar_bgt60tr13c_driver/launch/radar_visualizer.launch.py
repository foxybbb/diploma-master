from launch import LaunchDescription
from launch_ros.actions import Node  # Import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='radar_bgt60tr13c_driver',
            executable='radar_publisher_node',
            name='radar_publisher_node',
            output='screen',
        ),
        Node(
            package='radar_bgt60tr13c_driver',
            executable='radar_visualizer',
            name='radar_visualizer',
            output='screen',
        ),
    ])