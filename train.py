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

    TARGET_UPDATE_Frequency=300 #denotes the number of parameter updates of Qpolicy after which the target net is replaced with the policy net
    Gamma=0.95
    batch_size=128
    hist=[]


    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200 
    steps_done = 0
    frame_len=4
    buffer_size=8192

    memory=ReplayMemory(buffer_size)
    frame_holder=Frameholder(frame_len)


        
    from nes_py.wrappers import JoypadSpace
    import time
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT





    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env=MaxAndSkip(env,4)

    num_actions=env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ############################################################################
    Qtarget=Qfunction(num_inputChannel=frame_len,num_act=num_actions).double().to(device)
    Qpolicy=Qfunction(num_inputChannel=frame_len,num_act=num_actions).double().to(device)

    Qpolicy.load_state_dict(torch.load("model2.pt",map_location=torch.device('cpu')))
    Qtarget.load_state_dict(torch.load("model2.pt",map_location=torch.device('cpu')))

    Qtarget.load_state_dict(Qpolicy.state_dict())
    Qtarget.eval()


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
                Qpolicy.eval()
                ac=np.argmax(Qpolicy(state).cpu().detach().numpy())
                Qpolicy.train()
                return ac
        else:
            return random.randrange(num_act)
    #############################################################################

        
    optimizer = optim.SGD(Qpolicy.parameters(), lr=0.001, momentum=0.9)
    num_param_update=0



    death=False
    observation = env.reset()
    enough=frame_holder.push(observation.copy())
    for i_episode in range(1500):
        
        action_set=[]
        start_time=time.time()
        reward_count=0

        for t in count():
           # env.render()
            
            if enough:
                current_state=frame_holder.encode().unsqueeze(0).double().to(device)
                #observation_tensor=frame_holder.encode()
                action=select_action(current_state,num_actions)
                observation, reward, done, info = env.step(action)
                action_set.append(action)
                enough=frame_holder.push(observation.copy())
                new_state=frame_holder.encode().unsqueeze(0).double().to(device)


                if done:
                    observation = env.reset()
                    death=True
                if reward<-10:
                    death=True

                reward_count=reward_count+reward
                memory.push( (current_state,action,reward,new_state,death) )


                if len(memory) >= batch_size:
                    #print("Update start")
                    buffer=memory.sample(batch_size)
                    X_train,targets,action_mask=build_training_batch(buffer,num_actions,Qtarget,Gamma)
                    X_train=X_train.double().to(device)
                    targets=torch.from_numpy(targets).double().to(device)
                    action_mask=torch.from_numpy(action_mask).double().to(device)

                    optimizer.zero_grad()

                    outputs = Qpolicy(X_train)
                    loss = F.smooth_l1_loss(action_mask*outputs, action_mask*targets)

                    loss.backward()
                    num_param_update=num_param_update+1
                    #for param in Qpolicy.parameters():
                    #    param.grad.data.clamp_(-1, 1)

                    optimizer.step()
                    #print("Update end")

            else:
                observation, reward, done, info = env.step(env.action_space.sample())
                #action_set.append(action)
                enough=frame_holder.push(observation.copy())
                if done:
                    print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                    hist.append(t)
                    reward_count=0
                    frame_holder.empty()
                    enough=False
                    observation = env.reset()
                    enough=frame_holder.push(observation.copy())
                

            if death:
                death=False
                print("Episode {} finished after {} timesteps, Reward Sum: {} , time: {}".format(i_episode,t+1,reward_count,np.round(time.time()-start_time)))
                hist.append(t)
                reward_count=0
                frame_holder.empty()
                enough=False
                break
            
            
            if num_param_update > TARGET_UPDATE_Frequency:
                num_param_update=0
                Qtarget.load_state_dict(Qpolicy.state_dict())
                torch.save(Qpolicy.state_dict(), "model2.pt")










    env.close()
    # Qtarget.load_state_dict(Qpolicy.state_dict())
    torch.save(Qpolicy.state_dict(), "model.pt")
