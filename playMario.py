import gym
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import random
import torch.nn.functional as F
from gym import spaces
from PIL import Image
from torchvision import transforms
import cv2
import time
from model import Qfunction

#actions:
# 0: nothing
# 1: right
# 2: short jumb
# 3: run to the right
# 5: long jump
# 6: back

    

def makeOneHot(x,n):
    w=np.zeros((1,n))
    w[0,x]=1
    return w

def makeOneHotv2(s,x,n):
    w=np.zeros((1,n))
    w[0,s]=x
    return w

#            memory.push( (current_state,action,RR) )
#            memory.push( (current_state,action,reward,new_state,death) )

def build_training_batch(mm,num_ac,targetNet,gamma):
    X=[]
    actions=[]
    targets=[]
    for i in range(len(mm)):
        if i==0:
            X=mm[i][0]
        else:
            X=torch.cat( (X,mm[i][0]),0)
        
        if mm[i][4]:
            RR=mm[i][2]
        else: 
            new_statee=mm[i][3]
            RR=mm[i][2]+gamma*targetNet(new_statee).cpu().detach().numpy().max() # should not np.clip(RR,-15,15)
            
        targets.append(makeOneHotv2(mm[i][1],RR,num_ac))
        actions.append(makeOneHot(mm[i][1],num_ac))
        
    
    return X,np.array(targets).reshape(-1,num_ac),np.squeeze(actions)


    
    

def Preprocess(img):
    #torch.from_numpy(img[])
    mm=Image.fromarray(img[32:,:,:])
    mm=np.array(mm.resize((128,128)))
    gray = np.expand_dims(cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY),axis=-1)
    return torch.from_numpy(np.transpose(gray,[2,0,1]))


class Frameholder(object):
    def __init__(self, frame_len):
        # frame_len is the length or the number of frames used to encode a state, including the current step
        self.frame_len = frame_len
        self.memory = []

    def push(self, element):
        # elemnt will be an np.array of size(3,h,w)
        self.memory.append(element)
        if len(self.memory) > (self.frame_len):
            self.memory.pop(0)
            return True #meaning a complete sequence
        if len(self.memory) >=self.frame_len:
            return True
        return False
            

    def empty(self):
        self.memory=[]
    
    def encode(self,next_state=True):
        # the output will be of shape (len_frame,H,W)
        if len(self.memory)<self.frame_len:
            return -2
        
        current=self.memory  
        result=Preprocess(current[0])
        for i in range(1,len(current)):
            preprocessed=Preprocess(current[i])
            result=torch.cat( (result,preprocessed),0)
        return result
  

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class MaxAndSkip(gym.Wrapper):
   
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        flag=0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if reward==-15:
                flag=1
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
            
        if flag==1:
            total_reward=-15
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


    def reset(self):
        """gym.Env reset."""
        return self.env.reset()





if __name__ == "__main__":

    rand_action=0
    s=[]
    def select_action(state,num_act,eps=0.7):
        # goes from EPS_START to EPS_END
        # state should be a tensor of the current state
        global steps_done
        sample = random.random()
        #eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        #steps_done += 1
        eps_threshold=eps
        if sample > eps_threshold:
            with torch.no_grad():
                Qtest.eval()
                ac=np.argmax(Qtest(state).cpu().detach().numpy())
                return ac
        else:
            #rand_action=rand_action+1
            return random.randrange(num_act)

    #### Run an eposide to see how the agent performs the task
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_len=4

    from nes_py.wrappers import JoypadSpace
    import time
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env=MaxAndSkip(env,4)
    num_actions=env.action_space.n


    frame_holder=Frameholder(frame_len)
    Qtest=Qfunction(num_inputChannel=frame_len,num_act=num_actions).double().to(device)
    Qtest.load_state_dict(torch.load("model2.pt",map_location=torch.device('cpu')))
    Qtest.eval()
                          
    observation = env.reset()
    s.append(observation.copy())
    enough=frame_holder.push(observation.copy())
    actions=[]
    r=0
    for t in count():
            env.render()
            if enough:
                state=frame_holder.encode().unsqueeze(0).double().to(device)
    #             action=np.argmax(Qtest(state).cpu().detach().numpy())
                action=select_action(state,num_act=num_actions,eps=0.95)
            else:
                action=random.randrange(num_actions)
            
            actions.append(action)
            observation, reward, done, info = env.step(action)
            s.append(observation.copy())

            r=r+reward
            enough=frame_holder.push(observation.copy())
            if reward<-8:
                frame_holder.empty()
                enough=False
                print(r)
                r=0
                break
            if done:
                print("Finished after {} timesteps".format(t+1))
                print(r)
                r=0
                break


    env.close()






    img_array = []

    height, width, layers = s[0].shape
    size = (width,height)

    for i in range(len(s)):
        img_array.append(s[i])
     
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(cv2.cvtColor(np.array(img_array[i]), cv2.COLOR_RGB2BGR))

    out.release()













