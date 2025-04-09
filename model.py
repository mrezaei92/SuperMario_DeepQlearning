import gym
import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import cv2
import time
#actions:
# 0: nothing
# 1: right
# 2: short jumb
# 3: run to the right
# 5: long jump
# 6: back

class Qfunction(torch.nn.Module):
    def __init__(self,num_inputChannel=4,num_act=6):
        super(Qfunction, self).__init__()
        ## assumes the input spatial dimension is 128 by 128
        self.num_actions=num_act
        self.num_inputChannel=num_inputChannel
        self.features=nn.Sequential(nn.Conv2d(num_inputChannel, 32, kernel_size=5, stride=2, padding=2) ,nn.BatchNorm2d(32),     nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2) ,nn.BatchNorm2d(64),     nn.ReLU(),
               nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2) ,nn.BatchNorm2d(128),     nn.ReLU(),
               nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=2) ,nn.BatchNorm2d(128),     nn.ReLU(),
               nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=2) ,nn.BatchNorm2d(128) ,    nn.ReLU(),
               nn.AdaptiveMaxPool2d(2))
        
        self.action_extractor=nn.Sequential(
            nn.Linear(128*4, 512),nn.BatchNorm1d(512),nn.ReLU(True),nn.Dropout(p=0.2),
            nn.Linear(512, 128),nn.BatchNorm1d(128),nn.ReLU(True),nn.Dropout(p=0.1),
            nn.Linear(128, self.num_actions)
        )

        
    def forward(self, x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.action_extractor(x)
        return x


    

def makeOneHot(x,n):
    w=np.zeros((1,n))
    w[0,x]=1
    return w

def makeOneHotv2(s,x,n):
    w=np.zeros((1,n))
    w[0,s]=x
    return w

