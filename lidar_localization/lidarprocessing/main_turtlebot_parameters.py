# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data, qos_profile_system_default
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose,PoseStamped

from rcl_interfaces.srv import GetParameters


# def main(args=None):
#     rclpy.init(args=args)
#     node = rclpy.create_node('turtlebotParameters')
#     node.declare_parameters(
#         namespace='',
#         parameters=[
#             ('my_str', 1),
#             ('my_int', 2),
#             ('my_double_array', True)
#         ]
#     )
    
    
#     print("ready")
#     try:
#         # while True:
#             # rclpy.spin_once(node,timeout_sec=0.001)
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#      	pass
    
    
#     rclpy.shutdown()
    
    

# if __name__ == '__main__':
#     main()
    
params={}

params['REL_POS_THRESH']=0.5 # meters after which a keyframe is made
params['REL_ANGLE_THRESH']=15*np.pi/180
params['ERR_THRES']=2.5
params['n_components']=35
params['reg_covar']=0.002

params['BinDownSampleKeyFrame_dx']=0.05
params['BinDownSampleKeyFrame_probs']=0.1

params['Plot_BinDownSampleKeyFrame_dx']=0.05
params['Plot_BinDownSampleKeyFrame_probs']=0.001

params['doLoopClosure'] = True
params['doLoopClosureLong'] = True

params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=25
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.25,0.25],dtype=np.float64)
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.6
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.5
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 16
params['offsetNodesBy'] = 2
params['MAX_NODES_ADJ_COMBINE']=5

params['NearLoopClose'] = {}
params['NearLoopClose']['Method']='GMM'
params['NearLoopClose']['PoseGrid']=None #pt2dproc.getgridvec(np.linspace(-np.pi/12,np.pi/12,3),np.linspace(-1,1,3),np.linspace(-1,1,3))
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['Method'] = 'GMM'
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
params['LongLoopClose']['PoseGrid']= None
params['LongLoopClose']['isPoseGridOffset']=True
params['LongLoopClose']['isBruteForce']=False

# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4
params['#ThreadsLoopClose']=8

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1




class TurtleBotParameterService(Node):
    
    def __init__(self):
        super().__init__('turtlebot_paraameter_service')
        self.params={}

        self.params['REL_POS_THRESH']=0.5 # meters after which a keyframe is made
        self.params['ERR_THRES']=2.5
        self.params['n_components']=35
        self.params['reg_covar']=0.002
        
        self.params['PLOT_AFTER_#KEYFRAMES'] = 5 # currently not used
        
        
        self.params['doLoopClosure'] = True
        self.params['LOOP_CLOSURE_D_THES']=0.3
        self.params['LOOP_CLOSURE_POS_THES']=10
        self.params['LOOP_CLOSURE_POS_MIN_THES']=0.1
        self.params['LOOP_CLOSURE_ERR_THES']= 3
        self.params['LOOPCLOSE_BIN_MATCHER_dx'] = 1
        self.params['LOOPCLOSE_BIN_MATCHER_L'] = 7
        
        self.params['LOOPCLOSE_AFTER_#KEYFRAMES'] = 6
        
        self.params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.4
        
        self.params['Do_GMM_FINE_FIT']=True
        
        self.params['Do_BIN_FINE_FIT'] = True
        
        self.params['Do_BIN_DEBUG_PLOT-dx']=False
        self.params['Do_BIN_DEBUG_PLOT']= False
        
        self.params['xy_hess_inv_thres']=10000*0.4
        self.params['th_hess_inv_thres']=100000*0.4
        self.params['#ThreadsLoopClose']=4
        
        self.params['INTER_DISTANCE_BINS_max']=120
        self.params['INTER_DISTANCE_BINS_dx']=1

        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        
    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

    
    