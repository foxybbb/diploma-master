#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import struct

class RadarVisualizer(Node):
    def __init__(self):
        super().__init__('radar_visualizer')
        
        # Create subscription to radar data
        self.subscription = self.create_subscription(
            PointCloud2,
            'radar/pointcloud',
            self.radar_callback,
            10)
        
        # Initialize plot
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], c=[], cmap='viridis')
        self.ax.set_xlim(-5, 5)  # Adjust based on your radar's range
        self.ax.set_ylim(0, 5)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Radar Point Cloud')
        
        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()
        
        self.get_logger().info('Radar visualizer node started')
        
    def radar_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = np.frombuffer(msg.data, dtype=np.float32)
        points = points.reshape(-1, 4)  # Assuming x, y, z, intensity format
        
        # Store points for plotting
        self.points = points
        
    def update_plot(self, frame):
        if hasattr(self, 'points'):
            # Update scatter plot with new points
            self.scatter.set_offsets(self.points[:, :2])
            self.scatter.set_array(self.points[:, 3])  # Use intensity for color
            self.scatter.set_clim(0, 1)  # Normalize intensity range
        return self.scatter,

def main(args=None):
    rclpy.init(args=args)
    visualizer = RadarVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 