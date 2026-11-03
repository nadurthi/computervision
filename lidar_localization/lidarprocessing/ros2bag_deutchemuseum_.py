import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu, MultiEchoLaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import pickle
import signal
import datetime


class TopicSaver:
    def __init__(self):
        self.Lscan=[]
        self.Limu=[]
        self.Lodom=[]
        self.Lcmd=[]
        self.T=[]
        
    def save(self):
        print("Saving")
        with open("b2-2016-04-27-12-31-41.pkl",'wb') as fh:
            pickle.dump({'scan':self.Lscan,'imu':self.Limu,'odom':self.Lodom,'cmd_vel':self.Lcmd},fh)        
            
    def scan_listener_callback(self,msg):
        print('I heard scan')
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        
        print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        print("msg.angle_min,msg.angle_max = ",msg.angle_min,msg.angle_max)
        print("msg.range_min,msg.range_max = ",msg.range_min,msg.range_max)
        print("msg.angle_increment,msg.time_increment = ",msg.angle_increment,msg.time_increment)
        print("msg.scan_time = ",msg.scan_time)   
        
        
        L=[list(msg.ranges[i].echoes) for i in range(len(msg.ranges))]
        self.Lscan.append({'scan':L,'time':T})

        
    def imu_listener_callback(self,msg):
        print('I heard imu')
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Limu.append({'w':[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
    		'a':[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],
            'time': T 
    		})
    def odom_listener_callback(self,msg):
        print('I heard odom')
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Lodom.append({'trans':[msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z],
    		'q':[msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w],
            'time': T 
    		})
    def cmd_listener_callback(self,msg):
        print('I heard cmd')
        # T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        # print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Lcmd.append({'v':[msg.linear.x,msg.linear.y,msg.linear.z],
    		'Om':[msg.angular.x,msg.angular.y,msg.angular.z],
            # 'time': T 
    		})    

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotsubscriber')
    ts=TopicSaver()
    node.create_subscription(MultiEchoLaserScan,'horizontal_laser_2d',ts.scan_listener_callback)
    node.create_subscription(Imu,'imu',ts.imu_listener_callback)
    #node.create_subscription(Odometry,'odom',ts.odom_listener_callback)
    #node.create_subscription(Twist,'cmd_vel',ts.cmd_listener_callback)
    
    # minimal_subscriber = ScanSubscriber()
    # rclpy.get_default_context().on_shutdown(minimal_subscriber.save)
    # signal.signal(signal.SIGINT, minimal_subscriber.save)

    # rclpy.spin(minimal_subscriber)
    try:
    	rclpy.spin(node)
    except KeyboardInterrupt:
    	pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_subscriber.save()
    # minimal_subscriber.destroy_node()
    
    print("saving and closeing")
    ts.save()
    rclpy.shutdown()


if __name__ == '__main__':
    main()