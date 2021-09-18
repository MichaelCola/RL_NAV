# -*- coding: utf-8 -*-
from pickle import TRUE
from time import time
from numpy.core.numeric import False_
from environment_stage_5 import Env
from elegantrl import agent
from elegantrl.run import Arguments, train_and_evaluate
import rospy
import time
from rospy.client import FATAL


if __name__ == '__main__':
    rospy.init_node('ddpg_stage_5')
    env = Env()
    state = env.reset()
    Agent = agent.AgentPPO()

    #初始化训练参数
    arguments = Arguments(agent = Agent , env = env , gpu_id= 00.0 ,if_on_policy= True)   #ppo is on_policy
    # arguments.if_per = True 
    arguments.init_before_training()
    # time_ = str(time.time())
    #开始训练
    train_and_evaluate(arguments)



