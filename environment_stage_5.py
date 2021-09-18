#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

from numpy.lib.function_base import angle, average
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
world = False
if world:
    from respawnGoal_custom_worlds import Respawn
else:
    from respawnGoal import Respawn
import copy
target_not_movable = False

ACTION_V_MAX = 1. # m/s
ACTION_W_MAX = 1. # rad/s

class Env():
    def __init__(self, action_dim=2):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.stopped = 0
        self.action_dim = action_dim
        self.v_var_rate = 0
        self.w_var_rate = 0
        self.var_rate_limit = 0.1
        self.time_step = 0.2
        self.action_dim = 2 
        self.state_dim = 42
        self.if_discrete = False
        self.env_name = "stage_8"
        self.max_step = 200
        self.target_return = 1000
        self.last_rate = np.zeros(2)
        self.max_linda_distance = 3.5
        self.state = np.zeros(self.state_dim)
        self.weight = 0.7

        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        _, _, yaw = euler_from_quaternion(orientation_list)
        
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi 

        elif heading < -pi:
            heading += 2 * pi
        self.heading = round(heading, 3)

    def getState(self, scan, past_action):
        scan_range = []
        heading = self.heading
        min_range = 0.25
        done = False

        # self.last_state = copy.deepcopy(self.state)
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        a, b, c, d = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.x)), float('{0:.3f}'.format(self.position.y)), float('{0:.3f}'.format(self.past_position.y))
        if abs(a - b)<0.001 and abs(c -d)<0.001:
            # rospy.loginfo('\n<<<<<Stopped>>>>>\n')
            # print('\n' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n')
            self.stopped += 1
            if self.stopped == 200:
                rospy.loginfo('Robot is in the same  200 times in a row')
                self.stopped = 0
                done = True
        else:
            # rospy.loginfo('\n>>>>> not stopped>>>>>\n')
            self.stopped = 0

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)*30
        
        if min_range > min(scan_range) > 0:
            done = True

        for pa in past_action:
            scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        # current_distance = self.getGoalDistace()
        if current_distance < 0.2:
            self.get_goalbox = True
        
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle ],  done

    def setReward(self, state, done, action):
        current_distance = state[-3]
        heading = state[-4]
        reward = 0
        v_reward = 0                     #线速度修正奖励
        w_reward = 0                     #角速度修正奖励
        limlit_reward = 0                #线速度与角速度的配合奖励  ， 角速度很大时 ，线速度要小
        a = math.sqrt(2)/ pi             #限制 v_reward 在 -2 到 2 的取值内,是可调参数

        #以下部分可以简化，为了可读性方便理解便分开来写，v_reward 用 一元二次方程拟合 即 r = ((ax)^2 + c0) * b  
        if heading >=0 :
            if heading < (pi/2) :        #当偏航角小于 pi/2 时 ，若动作线速度是正数 ，则是偏向于靠近目标点的行为 。否则是远离目标点的行为
                if action[0] > 0 :
                    v_reward += (- ((a * heading)**2 ) + ((a * pi)**2) ) * ( action[0] / ACTION_V_MAX )
                else  : v_reward -= 1.5*(- ((a * heading)**2 ) + ((a * pi)**2) ) * ( -action[0] / ACTION_V_MAX )   #取1.5倍是因为防止前后移动相同距离的reward和为0

            elif heading > (pi/2) :     #当偏航角大于于 pi/2 时 ，若动作线速度是负数 ，则是偏向于靠近目标点的行为 。否则是远离目标点的行为
                if action[0] < 0 :
                    v_reward += 0.9*(- ((a * heading)**2 ) + ((a * pi)**2) )* ( -action[0] / ACTION_V_MAX )   #0.9倍抑制倒退
                else : v_reward -= 1.5*(- ((a * heading)**2 ) + ((a * pi)**2) )* ( action[0] / ACTION_V_MAX )
            else :                     #当偏航角等于 pi/2 时 ，无论正负都是远离目标点的行为
                v_reward -= 2
        else : 
            if heading > -(pi/2):
                if action[0] > 0:
                    v_reward += (- ((a * -heading)**2 ) + ((a* pi)**2) ) * ( action[0] / ACTION_V_MAX )    
                else : v_reward -= 1.5*(- ((a * -heading)**2 ) + ((a * pi)**2) ) * ( -action[0] / ACTION_V_MAX )
            elif heading < -(pi/2):
                if action[0] < 0:
                    v_reward += 0.9*(- ((a * -heading)**2 ) + ((a* pi)**2) ) * ( -action[0] / ACTION_V_MAX )    
                else :  v_reward -= 1.5*(- ((a * -heading)**2 ) + ((a * pi)**2) ) * ( action[0] / ACTION_V_MAX )   
            else : v_reward -= 2

        limlit_reward = 2*math.fabs(math.fabs(action[1]) - math.fabs(action[0]))  #提高线速度 和 角速度 的差值
        # 角速度的修正
        angle = heading + (pi / (ACTION_W_MAX*4) *(-action[1])) + pi 
        #正对目标奖励
        w_reward = round(2* (1 - math.fabs(-2/ pi * angle +2.)), 2)  
        #当距离大于 0.4 时 ， 线速度的奖励权重 > 角度的修正奖励权重  ，小于 0.4 时，权重相反
        if current_distance <=  0.4 :   
            weight = (1 - self.weight)
        else:  weight = copy.deepcopy(self.weight)
        reward = (v_reward * weight )+  (w_reward * (1- weight)) + limlit_reward
        if done:
            reward += -50
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            reward += 80
            self.pub_cmd_vel.publish(Twist())
            if world:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, running=True)
                if target_not_movable:
                    self.reset()
            else:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        # print("reward:",reward)
        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        self.v_var_rate = abs(past_action[0] - linear_vel)/self.time_step
        self.w_var_rate = abs(past_action[1] - ang_vel)/self.time_step
        self.last_rate= [copy.deepcopy(self.v_var_rate) , copy.deepcopy( self.w_var_rate)]
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done,  = self.getState(data, past_action)
        reward = self.setReward(state, done, past_action )

        return np.asarray(state), reward, done , None

    def reset(self):
        #print('aqui2_____________---')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        self.goal_distance = self.getGoalDistace()
        state, _ = self.getState(data, [0]*self.action_dim)

        return np.asarray(state)