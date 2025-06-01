from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='radar_robot_application',
            executable='robot_processing_node',
            name='robot_processing_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                # Add any parameters here if needed in the future
            }]
        )
    ])
