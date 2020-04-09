import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_UNITS_1 = 256*3
HIDDEN_UNITS_2 = 256*2
MAX_ACTION = 7000
b = 1.875e-7

class Actor(nn.Module):


    def __init__(self, state_size, action_size, seed=0, fc1_units=HIDDEN_UNITS_1, fc2_units=HIDDEN_UNITS_2):

        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, action_size)


    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        x = MAX_ACTION*x
        x = torch.pow(x,2)*b 
        return x

class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fcs1_units=HIDDEN_UNITS_1, fc2_units=HIDDEN_UNITS_2):
        super(Critic, self).__init__()
        
        
        #Arquitetura Q1
        self.l1 = nn.Linear(state_size+action_size, fcs1_units)
        self.l2 = nn.Linear(fcs1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)
        
        #Arquitetura Q2
        self.l4 = nn.Linear(state_size+action_size, fcs1_units)
        self.l5 = nn.Linear(fcs1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)        


    def forward(self, state, action):
        xu = torch.cat([state,action],1)
        
        x1 = torch.relu(self.l1(xu))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        
        x2 = torch.relu(self.l4(xu))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1,x2
    
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1