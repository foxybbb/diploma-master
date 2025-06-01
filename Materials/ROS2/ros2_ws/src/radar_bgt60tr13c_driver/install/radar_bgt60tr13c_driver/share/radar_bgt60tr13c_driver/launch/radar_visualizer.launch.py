from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='radar_bgt60tr13c_driver',
            executable='radar_visualizer.py',
            name='radar_visualizer',
            output='screen'
        )
    ])
