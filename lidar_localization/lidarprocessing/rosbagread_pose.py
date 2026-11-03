import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
import pickle
import signal
import datetime


class PoseSubscriber(Node):

    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.L=[]
        self.cnt=0

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%d"' % self.cnt)
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.L.append({
        'position':[msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z],
        'orientation':[msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w],
        'time': T 
		})
        self.cnt+=1

    def save(self):
    	print("Saving")
    	with open("houseScan_pose.pkl",'wb') as fh:
    		pickle.dump(self.L,fh)

def main(args=None):
    rclpy.init(args=args)

    
    pose_subscriber = PoseSubscriber()
    # rclpy.get_default_context().on_shutdown(minimal_subscriber.save)
    # signal.signal(signal.SIGINT, minimal_subscriber.save)

    # rclpy.spin(minimal_subscriber)
    try:
    	rclpy.spin(pose_subscriber)
    except KeyboardInterrupt:
    	pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_subscriber.save()
    pose_subscriber.destroy_node()
    

    rclpy.shutdown()


if __name__ == '__main__':
    main()