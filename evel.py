
from numpy.lib.npyio import load
import torch
import rospy
from environment_stage_5 import Env
import numpy as np 
from elegantrl import net
# import torchvision.models as models
from copy import deepcopy 





if __name__ == '__main__':
    rospy.init_node('ddpg_stage_5')
    env = Env()
    Agent = net.ActorSAC(mid_dim= 2**8 , state_dim= 42 , action_dim=2)
    Agent.load_state_dict(torch.load('actor.pth'))
    state = env.reset()
    past_action = np.zeros(2)
    while(True):
        state_tensor = torch.as_tensor((state,) , dtype=torch.float32 )
        a_tensor = Agent(state_tensor)
        action = a_tensor.detach().cpu().numpy()[0]
        print("action:",action)
        state, reward, done, _ = env.step(action,past_action)
        past_action = deepcopy(action)
        if done:
            state = env.reset()