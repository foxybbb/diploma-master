import rclpy
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
import time # Though get_clock().now().to_msg() is generally preferred for ROS time
import numpy as np # For OccupancyGrid

def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles to a Quaternion."""
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
        # Publishers for hoverboard feedback
        self.hb_vel_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/velocity", 10)
        self.hb_vel_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/velocity", 10)
        self.hb_pos_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/position", 10)
        self.hb_pos_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/position", 10)
        self.hb_curr_pub_left = self.create_publisher(Float64, "hoverboard/left_wheel/dc_current", 10)
        self.hb_curr_pub_right = self.create_publisher(Float64, "hoverboard/right_wheel/dc_current", 10)
        self.hb_voltage_pub = self.create_publisher(Float64, "hoverboard/battery_voltage", 10)
        self.hb_board_temp_pub = self.create_publisher(Float64, "hoverboard/temperature", 10) # Hoverboard temp
        self.hb_connected_pub = self.create_publisher(Bool, "hoverboard/connected", 10)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 50) # QoS 50 for odometry
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Robot command inputs
        self.left_wheel_cmd = 0.0  # Target wheel speed in rad/s
        self.right_wheel_cmd = 0.0 # Target wheel speed in rad/s
        self.hb_cmd_sub_left = self.create_subscription(
            Float64, "hoverboard/left_wheel/cmd", self.left_wheel_cmd_callback, 10)
        self.hb_cmd_sub_right = self.create_subscription(
            Float64, "hoverboard/right_wheel/cmd", self.right_wheel_cmd_callback, 10)

        # Robot physical parameters (dummy)
        self.wheel_base = 0.3  # meters (distance between wheels)
        self.wheel_radius = 0.05 # meters
        self.max_wheel_speed = 2.0 # rad/s, example maximum angular velocity of a wheel

        # Robot state for odom simulation - INITIALIZED HERE
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.sim_left_pos_rad = 0.0 # Accumulated position in radians
        self.sim_right_pos_rad = 0.0 # Accumulated position in radians

        # --- Radar Data Simulation ---
        self.get_logger().info("Initializing Radar Interface...")
        self.radar_iq_raw_pub = self.create_publisher(Float32MultiArray, "/bgt60/iq_raw", 10)
        self.radar_sensor_temp_pub = self.create_publisher(Temperature, "/bgt60/temp", 10) # Radar sensor temp
        self.radar_detections_pub = self.create_publisher(PointCloud2, "/bgt60/detections", 10)
        self.radar_tracks_pub = self.create_publisher(MarkerArray, "/radar/tracks", 10)

        # --- SLAM-like Simulation ---
        self.get_logger().info("Initializing SLAM-like Module...")
        # For /map, use a transient local QoS to ensure new subscribers get the last map
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            "/map",
            rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.map_resolution = 0.1  # meters/cell
        self.map_width_cells = 100  # cells
        self.map_height_cells = 100 # cells
        self.map_origin_x = - (self.map_width_cells / 2.0) * self.map_resolution
        self.map_origin_y = - (self.map_height_cells / 2.0) * self.map_resolution
        # Initialize map with unknown values (-1)
        self.occupancy_grid_data = np.full((self.map_height_cells, self.map_width_cells), -1, dtype=np.int8)
        self.publish_static_radar_tf() # base_link -> radar_link (done once)
        # map -> odom will be published by the timer, could be static or "drifting" for simulation

        # Timer for periodic updates
        self.timer_period = 0.1  # seconds (10Hz)
        self.timer = self.create_timer(self.timer_period, self.update_robot_state_and_publish)
        self.track_id_counter = 0
        self.map_publish_counter = 0 # To control map publish frequency

        self.get_logger().info('Robot_processing_node started.')

    def left_wheel_cmd_callback(self, msg):
        # Assuming msg.data is target rad/s for the wheel
        self.left_wheel_cmd = np.clip(msg.data, -self.max_wheel_speed, self.max_wheel_speed)
        # self.get_logger().info(f'Left CMD: {self.left_wheel_cmd:.2f} rad/s')


    def right_wheel_cmd_callback(self, msg):
        # Assuming msg.data is target rad/s for the wheel
        self.right_wheel_cmd = np.clip(msg.data, -self.max_wheel_speed, self.max_wheel_speed)
        # self.get_logger().info(f'Right CMD: {self.right_wheel_cmd:.2f} rad/s')


    def publish_map_to_odom_tf(self, stamp):
        # In a real SLAM, this transform would be dynamic.
        # For this simulation, we'll keep it static (map and odom origins aligned)
        # or introduce a very slow, predictable drift for testing.
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0 # No translation between map and odom origin
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = euler_to_quaternion(0,0,0) # No rotation
        self.tf_broadcaster.sendTransform(t)

    def publish_static_radar_tf(self):
        # This transform is static: base_link -> radar_link
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'radar_link'
        static_transform.transform.translation.x = 0.15  # Radar 0.15m ahead of base_link center
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.2  # Radar 0.2m above base_link center
        static_transform.transform.rotation = euler_to_quaternion(0, 0, 0) # No rotation relative to base_link
        self.static_tf_broadcaster.sendTransform(static_transform)
        self.get_logger().info('Published static TF: base_link -> radar_link')


    def update_robot_state_and_publish(self):
        current_time_msg = self.get_clock().now().to_msg()
        dt = self.timer_period

        # --- 1. Simulate Hoverboard Feedback & Update Odometry ---
        # Actual wheel velocities (for feedback) could have some noise or delay from cmd
        current_left_wheel_vel = self.left_wheel_cmd * random.uniform(0.95, 1.05)
        current_right_wheel_vel = self.right_wheel_cmd * random.uniform(0.95, 1.05)

        # Convert wheel angular velocities (rad/s) to linear/angular velocity of robot
        v_left_wheel_linear = current_left_wheel_vel * self.wheel_radius
        v_right_wheel_linear = current_right_wheel_vel * self.wheel_radius

        linear_velocity_robot = (v_right_wheel_linear + v_left_wheel_linear) / 2.0
        angular_velocity_robot = (v_right_wheel_linear - v_left_wheel_linear) / self.wheel_base

        # Update robot pose based on calculated velocities
        delta_x = linear_velocity_robot * math.cos(self.theta) * dt
        delta_y = linear_velocity_robot * math.sin(self.theta) * dt
        delta_theta = angular_velocity_robot * dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta)) # Normalize angle to [-pi, pi]

        # Update simulated wheel positions (in radians)
        self.sim_left_pos_rad += current_left_wheel_vel * dt
        self.sim_right_pos_rad += current_right_wheel_vel * dt

        # Publish hoverboard feedback
        msg_f64 = Float64()
        msg_f64.data = current_left_wheel_vel # Actual simulated rad/s
        self.hb_vel_pub_left.publish(msg_f64)
        msg_f64.data = current_right_wheel_vel
        self.hb_vel_pub_right.publish(msg_f64)

        msg_f64.data = self.sim_left_pos_rad # rad
        self.hb_pos_pub_left.publish(msg_f64)
        msg_f64.data = self.sim_right_pos_rad # rad
        self.hb_pos_pub_right.publish(msg_f64)

        msg_f64.data = random.uniform(0.1, 1.0) * abs(current_left_wheel_vel) # Dummy current
        self.hb_curr_pub_left.publish(msg_f64)
        msg_f64.data = random.uniform(0.1, 1.0) * abs(current_right_wheel_vel)
        self.hb_curr_pub_right.publish(msg_f64)

        msg_f64.data = 36.5 - abs(linear_velocity_robot) * 0.5 # Dummy voltage drop
        self.hb_voltage_pub.publish(msg_f64)
        msg_f64.data = 28.0 + abs(linear_velocity_robot) * 3 # Dummy temp increase
        self.hb_board_temp_pub.publish(msg_f64)
        msg_bool = Bool(); msg_bool.data = True; self.hb_connected_pub.publish(msg_bool)


        # Publish Odometry Message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time_msg
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position = Point(x=self.x, y=self.y, z=0.0)
        odom_msg.pose.pose.orientation = euler_to_quaternion(0, 0, self.theta)
        odom_msg.twist.twist.linear = Vector3(x=linear_velocity_robot, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=angular_velocity_robot)
        # Add covariance if needed for a more realistic simulation
        self.odom_pub.publish(odom_msg)

        # Publish odom -> base_link transform
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = current_time_msg
        t_odom_base.header.frame_id = 'odom'
        t_odom_base.child_frame_id = 'base_link'
        t_odom_base.transform.translation = Vector3(x=self.x, y=self.y, z=0.0)
        t_odom_base.transform.rotation = odom_msg.pose.pose.orientation # Use same quaternion
        self.tf_broadcaster.sendTransform(t_odom_base)

        # --- 2. Simulate and Publish Radar Data ---
        self.publish_dummy_radar_data(current_time_msg, 'radar_link', 'odom')

        # --- 3. Simulate SLAM-like Map Update and Publish ---
        self.update_and_publish_dummy_map(current_time_msg)
        self.publish_map_to_odom_tf(current_time_msg) # Publish map -> odom TF


    def publish_dummy_radar_data(self, stamp, radar_frame_id, tracks_odom_frame_id):
        # IQ Raw
        iq_msg = Float32MultiArray()
        iq_msg.layout.data_offset = 0
        iq_msg.layout.dim.append(MultiArrayDimension(label="channels", size=3, stride=3*128))
        iq_msg.layout.dim.append(MultiArrayDimension(label="samples", size=128, stride=128))
        iq_msg.data = [random.uniform(-0.5, 0.5) for _ in range(3 * 128)] # Smaller range for IQ
        self.radar_iq_raw_pub.publish(iq_msg)

        # Temperature
        temp_msg = Temperature()
        temp_msg.header.stamp = stamp
        temp_msg.header.frame_id = radar_frame_id
        temp_msg.temperature = random.uniform(38.0, 42.0) # Radar electronics temp
        temp_msg.variance = 0.01
        self.radar_sensor_temp_pub.publish(temp_msg)

        # Detections (PointCloud2 in radar_frame_id)
        points_bytes = bytearray()
        num_detections = random.randint(2, 8)
        for _ in range(num_detections):
            # Simulate detections in front of the radar
            r = random.uniform(0.3, 8.0) # range
            angle = random.uniform(-math.pi/3, math.pi/3) # +/- 60 degrees FOV
            x_radar = r * math.cos(angle)
            y_radar = r * math.sin(angle)
            z_radar = random.uniform(-0.05, 0.05) # Small Z variation in radar frame
            intensity = random.uniform(20.0, 150.0)
            points_bytes.extend(struct.pack('ffff', x_radar, y_radar, z_radar, intensity))

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
        pc2_msg.point_step = 16 # 4 floats * 4 bytes/float
        pc2_msg.row_step = pc2_msg.point_step * num_detections
        pc2_msg.data = bytes(points_bytes)
        pc2_msg.is_dense = True
        self.radar_detections_pub.publish(pc2_msg)

        # Tracks (MarkerArray in tracks_odom_frame_id, e.g., 'odom')
        marker_array_msg = MarkerArray()
        num_tracks = random.randint(0, 3)
        for i in range(num_tracks):
            track_id = (self.track_id_counter + i) % 15 # Keep IDs bounded

            # Simulate a track detected by radar, relative to robot, then transform to odom
            # Track position relative to radar_link frame
            track_r_radar = random.uniform(1.0, 6.0)
            track_angle_radar = random.uniform(-math.pi/3.5, math.pi/3.5) # Within radar FOV
            track_x_radar = track_r_radar * math.cos(track_angle_radar)
            track_y_radar = track_r_radar * math.sin(track_angle_radar)

            # Simple transform from radar_link to base_link (using static TF values)
            # radar_link to base_link: x_base = x_radar + 0.15 (static_tf.transform.translation.x)
            track_x_base = track_x_radar + 0.15
            track_y_base = track_y_radar # Assuming y is aligned

            # Transform from base_link to odom (using current robot pose)
            track_x_odom = self.x + track_x_base * math.cos(self.theta) - track_y_base * math.sin(self.theta)
            track_y_odom = self.y + track_x_base * math.sin(self.theta) + track_y_base * math.cos(self.theta)

            marker = Marker()
            marker.header.frame_id = tracks_odom_frame_id
            marker.header.stamp = stamp
            marker.ns = "radar_tracks_arrows"
            marker.id = track_id
            marker.type = Marker.ARROW; marker.action = Marker.ADD
            marker.pose.position = Point(x=track_x_odom, y=track_y_odom, z=0.15) # Height of track
            # Dummy velocity for arrow direction (in odom frame, could be more complex)
            vx_odom = random.uniform(-0.3, 0.3); vy_odom = random.uniform(-0.3, 0.3)
            angle_track_odom = math.atan2(vy_odom, vx_odom)
            marker.pose.orientation = euler_to_quaternion(0,0,angle_track_odom)
            marker.scale = Vector3(x=(math.sqrt(vx_odom**2+vy_odom**2)*0.8+0.2), y=0.15, z=0.15) # Arrow size
            marker.color.a=0.8; marker.color.r=0.1; marker.color.g=0.9; marker.color.b=0.1 # Greenish
            marker.lifetime = rclpy.duration.Duration(seconds=self.timer_period * 3.5).to_msg() # Persist for a few cycles
            marker_array_msg.markers.append(marker)

        self.track_id_counter += num_tracks # Increment base for next cycle
        self.radar_tracks_pub.publish(marker_array_msg)


    def update_and_publish_dummy_map(self, stamp):
        self.map_publish_counter +=1
        # Update map less frequently than other data to save processing if map is large
        if self.map_publish_counter % 20 == 0: # Publish map every 2 seconds (20 * 0.1s)
            # Simulate adding some "detected" obstacles to the map
            # This is a very crude "SLAM update" based on current odom and a fake detection
            if random.random() > 0.3: # 70% chance to add a new "wall" segment
                # Obstacle in odom frame (relative to current robot pose for demo)
                # Make obstacle appear somewhat in front of where the robot is heading
                dist_to_obs = random.uniform(1.0, 3.0)
                angle_offset = random.uniform(-math.pi/8, math.pi/8)
                obs_x_odom = self.x + dist_to_obs * math.cos(self.theta + angle_offset)
                obs_y_odom = self.y + dist_to_obs * math.sin(self.theta + angle_offset)

                # Convert odom coordinates to map grid cell
                # Assuming map->odom TF means map_origin is (0,0) in map frame,
                # and odom_origin is also (0,0) in odom frame, and they are aligned.
                # The map data itself uses its own origin.
                map_x_cell = int((obs_x_odom - self.map_origin_x) / self.map_resolution)
                map_y_cell = int((obs_y_odom - self.map_origin_y) / self.map_resolution)

                if 0 <= map_x_cell < self.map_width_cells and 0 <= map_y_cell < self.map_height_cells:
                    # Mark a small area around the detection as occupied
                    for r_offset in range(-1, 2): # 3x3 block
                        for c_offset in range(-1, 2):
                            final_y_cell, final_x_cell = map_y_cell + r_offset, map_x_cell + c_offset
                            if 0 <= final_x_cell < self.map_width_cells and 0 <= final_y_cell < self.map_height_cells:
                                self.occupancy_grid_data[final_y_cell, final_x_cell] = 100 # Occupied

            map_msg = OccupancyGrid()
            map_msg.header.stamp = stamp
            map_msg.header.frame_id = 'map'
            map_msg.info.map_load_time = stamp # Or actual load time if loading from file
            map_msg.info.resolution = self.map_resolution
            map_msg.info.width = self.map_width_cells
            map_msg.info.height = self.map_height_cells
            map_msg.info.origin.position.x = self.map_origin_x
            map_msg.info.origin.position.y = self.map_origin_y
            map_msg.info.origin.position.z = 0.0
            map_msg.info.origin.orientation = euler_to_quaternion(0,0,0) # Map origin orientation
            map_msg.data = self.occupancy_grid_data.ravel().tolist() # Flatten and convert to list
            self.map_pub.publish(map_msg)
            self.get_logger().info(f"Published dummy map update. Robot at ({self.x:.2f}, {self.y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = RobotProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("RobotProcessingNode shutting down...")
    finally:
        # It's good practice to destroy the node explicitly
        # though Python's garbage collector will eventually do it
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
