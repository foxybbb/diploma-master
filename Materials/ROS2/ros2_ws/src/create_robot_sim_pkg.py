import os
import stat
import math
import random
import struct
import time # For PointCloud2 header

# --- File Contents ---

PACKAGE_XML_REAL_CONTENT = """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>radar_robot_application</name>
  <version>0.0.1</version>
  <description>ROS2 package simulating radar processing, robot control, and SLAM-like interactions</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""

SETUP_PY_REAL_CONTENT = """from setuptools import find_packages, setup
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
"""

SETUP_CFG_REAL_CONTENT = """[develop]
script_dir=$base/lib/radar_robot_application
[install]
install_scripts=$base/lib/radar_robot_application
"""

ROBOT_PROCESSING_NODE_PY_CONTENT = """import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Temperature, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import random
import math
import struct
import time
import numpy as np # For OccupancyGrid

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

class RobotProcessingNode(Node):
    def __init__(self):
        super().__init__('robot_processing_node')

        # --- Robot Control & Odometry Simulation ---
        self.get_logger().info("Initializing Robot Control and Odometry...")
        self.hb_vel_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/velocity", 10)
        self.hb_vel_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/velocity", 10)
        # ... (other hoverboard publishers as before)
        self.hb_pos_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/position", 10)
        self.hb_pos_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/position", 10)
        self.hb_curr_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/dc_current", 10)
        self.hb_curr_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/dc_current", 10)
        self.hb_voltage_pub = self.create_publisher(Float64, "hoverboard/battery_voltage", 10)
        self.hb_board_temp_pub = self.create_publisher(Float64, "hoverboard/temperature", 10)
        self.hb_connected_pub = self.create_publisher(Bool, "hoverboard/connected", 10)


        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.left_wheel_cmd = 0.0
        self.right_wheel_cmd = 0.0
        self.hb_cmd_sub_left = self.create_subscription(
            Float64, "hoverboard/left_wheel/cmd", self.left_wheel_cmd_callback, 10)
        self.hb_cmd_sub_right = self.create_subscription(
            Float64, "hoverboard/right_wheel/cmd", self.right_wheel_cmd_callback, 10)

        # Robot physical parameters (dummy)
        self.wheel_base = 0.3  # meters
        self.wheel_radius = 0.05 # meters
        self.max_wheel_speed = 1.5 # rad/s, example

        # Robot state for odom simulation
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.sim_left_pos = 0.0
        self.sim_right_pos = 0.0


        # --- Radar Data Simulation ---
        self.get_logger().info("Initializing Radar Interface...")
        self.radar_iq_raw_pub = self.create_publisher(Float32MultiArray, "/bgt60/iq_raw", 10)
        self.radar_sensor_temp_pub = self.create_publisher(Temperature, "/bgt60/temp", 10)
        self.radar_detections_pub = self.create_publisher(PointCloud2, "/bgt60/detections", 10)
        self.radar_tracks_pub = self.create_publisher(MarkerArray, "/radar/tracks", 10)

        # --- SLAM-like Simulation ---
        self.get_logger().info("Initializing SLAM-like Module...")
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10) # QoS for map is usually latched
        self.map_resolution = 0.1  # meters/cell
        self.map_width = 100  # cells
        self.map_height = 100 # cells
        self.map_origin_x = - (self.map_width / 2.0) * self.map_resolution
        self.map_origin_y = - (self.map_height / 2.0) * self.map_resolution
        self.occupancy_grid_data = np.full((self.map_height, self.map_width), -1, dtype=np.int8) # -1 for unknown
        self.publish_static_map_tf() # map -> odom (initially identity or fixed)
        self.publish_static_radar_tf() # base_link -> radar_link

        # Timer for periodic updates
        self.timer_period = 0.1  # seconds (10Hz)
        self.timer = self.create_timer(self.timer_period, self.update_robot_state_and_publish)
        self.track_id_counter = 0
        self.map_publish_counter = 0

        self.get_logger().info('Robot_processing_node started.')

    def left_wheel_cmd_callback(self, msg):
        # Simulate speed command, e.g., msg.data could be target rad/s or normalized effort
        self.left_wheel_cmd = np.clip(msg.data, -self.max_wheel_speed, self.max_wheel_speed)

    def right_wheel_cmd_callback(self, msg):
        self.right_wheel_cmd = np.clip(msg.data, -self.max_wheel_speed, self.max_wheel_speed)

    def publish_static_map_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = euler_to_quaternion(0,0,0) # Initially map and odom are aligned
        self.tf_broadcaster.sendTransform(t) # SLAM would update this dynamically

    def publish_static_radar_tf(self):
        # This transform is static: base_link -> radar_link
        # Assuming radar is 0.2m forward on the robot's X-axis
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'radar_link'
        static_transform.transform.translation.x = 0.2  # Radar 0.2m ahead of base_link center
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.1  # Radar 0.1m above base_link center
        static_transform.transform.rotation = euler_to_quaternion(0, 0, 0) # No rotation relative to base_link
        self.static_tf_broadcaster.sendTransform(static_transform)


    def update_robot_state_and_publish(self):
        current_time = self.get_clock().now().to_msg()
        dt = self.timer_period

        # --- 1. Simulate Hoverboard Feedback & Update Odometry ---
        # Convert wheel commands (rad/s) to linear/angular velocity of robot
        v_left_wheel = self.left_wheel_cmd * self.wheel_radius
        v_right_wheel = self.right_wheel_cmd * self.wheel_radius

        linear_velocity = (v_right_wheel + v_left_wheel) / 2.0
        angular_velocity = (v_right_wheel - v_left_wheel) / self.wheel_base

        # Update robot pose
        delta_x = linear_velocity * math.cos(self.theta) * dt
        delta_y = linear_velocity * math.sin(self.theta) * dt
        delta_theta = angular_velocity * dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta)) # Normalize angle

        self.sim_left_pos += self.left_wheel_cmd * dt
        self.sim_right_pos += self.right_wheel_cmd * dt

        # Publish hoverboard feedback
        msg_f64 = Float64()
        msg_f64.data = self.left_wheel_cmd # Assuming cmd is directly translatable to measured velocity for dummy
        self.hb_vel_pub_left.publish(msg_f64)
        msg_f64.data = self.right_wheel_cmd
        self.hb_vel_pub_right.publish(msg_f64)
        msg_f64.data = self.sim_left_pos
        self.hb_pos_pub_left.publish(msg_f64)
        msg_f64.data = self.sim_right_pos
        self.hb_pos_pub_right.publish(msg_f64)
        # ... (publish other dummy hoverboard feedback: current, voltage, temp, connected)
        msg_f64.data = random.uniform(0.1, 1.0) * abs(self.left_wheel_cmd)
        self.hb_curr_pub_left.publish(msg_f64)
        msg_f64.data = random.uniform(0.1, 1.0) * abs(self.right_wheel_cmd)
        self.hb_curr_pub_right.publish(msg_f64)
        msg_f64.data = 36.0 - abs(linear_velocity) # Voltage drop under load
        self.hb_voltage_pub.publish(msg_f64)
        msg_f64.data = 30.0 + abs(linear_velocity) * 5 # Temp increase under load
        self.hb_board_temp_pub.publish(msg_f64)
        msg_bool = Bool(); msg_bool.data = True; self.hb_connected_pub.publish(msg_bool)


        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position = Point(x=self.x, y=self.y, z=0.0)
        odom_msg.pose.pose.orientation = euler_to_quaternion(0, 0, self.theta)
        odom_msg.twist.twist.linear = Vector3(x=linear_velocity, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=angular_velocity)
        # Add covariance if needed
        self.odom_pub.publish(odom_msg)

        # Publish odom -> base_link transform
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = current_time
        t_odom_base.header.frame_id = 'odom'
        t_odom_base.child_frame_id = 'base_link'
        t_odom_base.transform.translation = Vector3(x=self.x, y=self.y, z=0.0)
        t_odom_base.transform.rotation = euler_to_quaternion(0, 0, self.theta)
        self.tf_broadcaster.sendTransform(t_odom_base)
        
        # --- 2. Simulate and Publish Radar Data ---
        # (Using radar_link as frame_id for sensor data, base_link for tracks in odom)
        self.publish_dummy_radar_data(current_time, 'radar_link', 'odom')

        # --- 3. Simulate SLAM-like Map Update and Publish ---
        self.update_and_publish_dummy_map(current_time)
        # SLAM would also update map->odom TF, here it's static after init or simple drift
        # For a more dynamic dummy SLAM, you could make map->odom drift slightly
        # self.publish_static_map_tf() # Or a slightly drifting one


    def publish_dummy_radar_data(self, stamp, radar_frame_id, tracks_frame_id):
        # IQ Raw
        # IQ Raw
        iq_msg = Float32MultiArray()
        iq_msg.layout.data_offset = 0 # CORRECT: data_offset is part of MultiArrayLayout
        # CORRECT: Removed data_offset from MultiArrayDimension constructor
        iq_msg.layout.dim.append(MultiArrayDimension(label="channels", size=3, stride=3*128))
        iq_msg.layout.dim.append(MultiArrayDimension(label="samples", size=128, stride=128))
        iq_msg.data = [random.uniform(-1.0, 1.0) for _ in range(3 * 128)]
        self.radar_iq_raw_pub.publish(iq_msg)

        # Temperature
        temp_msg = Temperature()
        temp_msg.header.stamp = stamp
        temp_msg.header.frame_id = radar_frame_id
        temp_msg.temperature = random.uniform(35.0, 45.0)
        temp_msg.variance = 0.01
        self.radar_sensor_temp_pub.publish(temp_msg)

        # Detections (PointCloud2 in radar_frame_id)
        points_bytes = bytearray()
        num_detections = random.randint(1, 5)
        for _ in range(num_detections):
            # Simulate detections in front of the radar
            r = random.uniform(0.5, 5.0) # range
            angle = random.uniform(-math.pi/4, math.pi/4) # +/- 45 degrees FOV
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = random.uniform(-0.1, 0.1)
            intensity = random.uniform(10.0, 100.0)
            points_bytes.extend(struct.pack('ffff', x, y, z, intensity))

        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = stamp
        pc2_msg.header.frame_id = radar_frame_id # Detections are in radar's own frame
        pc2_msg.height = 1
        pc2_msg.width = num_detections
        pc2_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * num_detections
        pc2_msg.data = bytes(points_bytes)
        pc2_msg.is_dense = True
        self.radar_detections_pub.publish(pc2_msg)

        # Tracks (MarkerArray in tracks_frame_id, e.g., 'odom' or 'map')
        marker_array_msg = MarkerArray()
        num_tracks = random.randint(0, 2)
        for i in range(num_tracks):
            track_id = (self.track_id_counter + i) % 20
            
            # Simulate a track moving relative to the odom frame
            # Position of track in odom frame
            track_x_odom = self.x + random.uniform(1.0, 4.0) * math.cos(self.theta + random.uniform(-0.5, 0.5))
            track_y_odom = self.y + random.uniform(1.0, 4.0) * math.sin(self.theta + random.uniform(-0.5, 0.5))
            
            marker = Marker()
            marker.header.frame_id = tracks_frame_id
            marker.header.stamp = stamp
            marker.ns = "radar_tracks_arrows"
            marker.id = track_id
            marker.type = Marker.ARROW; marker.action = Marker.ADD
            marker.pose.position = Point(x=track_x_odom, y=track_y_odom, z=0.2)
            # Dummy velocity for arrow direction
            vx = random.uniform(-0.5, 0.5); vy = random.uniform(-0.2, 0.2)
            angle_track = math.atan2(vy, vx)
            marker.pose.orientation = euler_to_quaternion(0,0,angle_track)
            marker.scale = Vector3(x=(math.sqrt(vx**2+vy**2)*0.5+0.2), y=0.1, z=0.1)
            marker.color.a=1.0; marker.color.r=0.0; marker.color.g=1.0; marker.color.b=0.0
            marker.lifetime = rclpy.duration.Duration(seconds=self.timer_period * 5).to_msg()
            marker_array_msg.markers.append(marker)
        self.track_id_counter += num_tracks
        self.radar_tracks_pub.publish(marker_array_msg)


    def update_and_publish_dummy_map(self, stamp):
        # Simulate adding some "detected" obstacles to the map based on radar tracks
        # This is a very crude simulation of mapping
        self.map_publish_counter +=1
        if self.map_publish_counter % 10 == 0: # Update map less frequently
            # For any "track" detected (use last cycle's simulated tracks for simplicity)
            # This part would use actual radar_tracks data if it were a real SLAM
            num_new_obstacles = random.randint(0,2)
            for _ in range(num_new_obstacles):
                # Obstacle in odom frame (relative to current robot pose for demo)
                obs_x_odom = self.x + random.uniform(1.0, 3.0) * math.cos(self.theta + random.uniform(-math.pi/6, math.pi/6))
                obs_y_odom = self.y + random.uniform(1.0, 3.0) * math.sin(self.theta + random.uniform(-math.pi/6, math.pi/6))
                
                # Convert odom coordinates to map grid cell
                # (Assuming map->odom is identity for this dummy map update logic)
                map_x = int((obs_x_odom - self.map_origin_x) / self.map_resolution)
                map_y = int((obs_y_odom - self.map_origin_y) / self.map_resolution)

                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    # Mark a small area around the detection as occupied
                    for r_offset in range(-1, 2):
                        for c_offset in range(-1, 2):
                            final_y, final_x = map_y + r_offset, map_x + c_offset
                            if 0 <= final_x < self.map_width and 0 <= final_y < self.map_height:
                                self.occupancy_grid_data[final_y, final_x] = 100 # Occupied
            
            map_msg = OccupancyGrid()
            map_msg.header.stamp = stamp
            map_msg.header.frame_id = 'map'
            map_msg.info.resolution = self.map_resolution
            map_msg.info.width = self.map_width
            map_msg.info.height = self.map_height
            map_msg.info.origin.position.x = self.map_origin_x
            map_msg.info.origin.position.y = self.map_origin_y
            map_msg.info.origin.position.z = 0.0
            map_msg.info.origin.orientation = euler_to_quaternion(0,0,0)
            map_msg.data = self.occupancy_grid_data.flatten().tolist()
            self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("RobotProcessingNode shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

LAUNCH_FILE_REAL_CONTENT = """from launch import LaunchDescription
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
"""

def create_ros2_real_sim_package(base_src_path):
    package_name = "radar_robot_application"
    package_path = os.path.join(base_src_path, package_name)

    inner_module_dir_path = os.path.join(package_path, package_name)
    launch_dir_path = os.path.join(package_path, "launch")
    resource_dir_path = os.path.join(package_path, "resource")

    dirs_to_create = [
        package_path, inner_module_dir_path, launch_dir_path, resource_dir_path
    ]
    print(f"Creating package structure for '{package_name}' in '{base_src_path}'...")
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True); print(f"  Created directory: {dir_path}")

    files_to_create_info = [
        ("package.xml", PACKAGE_XML_REAL_CONTENT, package_path),
        ("setup.py", SETUP_PY_REAL_CONTENT, package_path),
        ("setup.cfg", SETUP_CFG_REAL_CONTENT, package_path),
        ("__init__.py", "", inner_module_dir_path),
        ("robot_processing_node.py", ROBOT_PROCESSING_NODE_PY_CONTENT, inner_module_dir_path),
        ("radar_robot_application_launch.py", LAUNCH_FILE_REAL_CONTENT, launch_dir_path), # Renamed launch file
        (package_name, "", resource_dir_path)
    ]
    for filename, content, dir_path_for_file in files_to_create_info:
        file_abs_path = os.path.join(dir_path_for_file, filename)
        try:
            with open(file_abs_path, "w") as f: f.write(content)
            print(f"  Created file: {file_abs_path}")
            if filename.endswith(".py"):
                 os.chmod(file_abs_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                 print(f"    Made executable: {file_abs_path}")
        except Exception as e: print(f"  Error creating file {file_abs_path}: {e}")

    print("\nPackage creation complete!")
    print("\nNext steps:")
    print(f"1. cd {os.path.dirname(os.path.abspath(base_src_path))}")
    print(f"2. colcon build --packages-select {package_name}")
    print(f"3. source install/setup.bash")
    print(f"4. ros2 launch {package_name} radar_robot_application_launch.py")
    print(f"5. In other terminals (after sourcing):")
    print(f"   ros2 topic echo /odom")
    print(f"   ros2 topic echo /map")
    print(f"   ros2 topic echo /radar/tracks")
    print(f"   ros2 topic pub /hoverboard/left_wheel/cmd std_msgs/msg/Float64 \"{{data: 0.5}}\" --once")
    print(f"   ros2 topic pub /hoverboard/right_wheel/cmd std_msgs/msg/Float64 \"{{data: 0.3}}\" --once")
    print(f"   rqt_graph")
    print(f"   rviz2 (Add TF, Odometry, PointCloud2, MarkerArray, OccupancyGrid displays. Set Fixed Frame to 'map' or 'odom')")

if __name__ == "__main__":
    default_ws_src_path = os.path.join(os.getcwd(), "ros2_radar_ws", "src")
    ws_src_path_input = input(f"Enter path to ROS2 workspace's 'src' (default: {default_ws_src_path}): ")
    ws_src_path_to_use = os.path.abspath(ws_src_path_input if ws_src_path_input else default_ws_src_path)
    if not os.path.exists(ws_src_path_to_use):
        print(f"Path '{ws_src_path_to_use}' DNE. Creating...")
        try: os.makedirs(ws_src_path_to_use, exist_ok=True); print(f"OK: {ws_src_path_to_use}")
        except OSError as e: print(f"ERR: {ws_src_path_to_use}. {e}"); exit()
    elif not os.path.isdir(ws_src_path_to_use): print(f"ERR: '{ws_src_path_to_use}' not dir."); exit()
    create_ros2_real_sim_package(ws_src_path_to_use)
