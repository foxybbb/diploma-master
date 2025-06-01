from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'radar_robot_application'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'numpy'], # Added numpy for occupancy grid
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='ROS2 package for radar_robot_application',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_processing_node = radar_robot_application.robot_processing_node:main',
        ],
    },
)
