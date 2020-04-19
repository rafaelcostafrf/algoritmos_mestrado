import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#NEURONIOS NAS CAMADAS INTERNAS
HU = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, HU)
        self.l2 = nn.Linear(HU, HU)
        self.l3 = nn.Linear(HU, HU)
        self.l4 = nn.Linear(HU, 4)
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = self.l4(a)
        return self.max_action * torch.sigmoid(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):      
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1_s = nn.Linear(state_dim,HU)
        self.l2_s = nn.Linear(HU,HU)
        self.l3_s = nn.Linear(HU,HU)
        
        self.l1_a = nn.Linear(action_dim, HU)
        self.l2_a = nn.Linear(HU,HU)
        self.l3_a = nn.Linear(HU,HU)
        
        self.l1 = nn.Linear(2*HU,1)

        
        # Q2 architecture
        self.l4_s = nn.Linear(state_dim,HU)
        self.l5_s = nn.Linear(HU,HU)
        self.l6_s = nn.Linear(HU,HU)
        
        self.l4_a = nn.Linear(action_dim, HU)
        self.l5_a = nn.Linear(HU,HU)
        self.l6_a = nn.Linear(HU,HU)
        
        
        self.l3 = nn.Linear(2*HU,1)


    def forward(self, state, action):
        #Q1
        q1_s = F.relu(self.l1_s(state))
        q1_s = F.relu(self.l2_s(q1_s))
        q1_s = self.l3_s(q1_s)
        
        q1_a = self.l1_a(action)
        q1_a = self.l2_a(q1_a)
        q1_a = self.l3_a(q1_a)
        
        q1 = torch.cat((q1_s,q1_a),1)
        q1 = F.relu(self.l1(q1))
        
        #Q2
        q2_s = F.relu(self.l4_s(state))
        q2_s = F.relu(self.l5_s(q2_s))
        q2_s = self.l6_s(q2_s)
        
        q2_a = self.l4_a(action)
        q2_a = self.l5_a(q2_a)
        q2_a = self.l6_a(q2_a)
        
        q2 = torch.cat((q2_s,q2_a),1)
        q2 = F.relu(self.l3(q2))
        return q1, q2


    def Q1(self, state, action):
        q1_s = F.relu(self.l1_s(state))
        q1_s = F.relu(self.l2_s(q1_s))
        q1_s = self.l3_s(q1_s)
        
        q1_a = self.l1_a(action)
        q1_a = self.l2_a(q1_a)
        q1_a = self.l3_a(q1_a)
        
        q1 = torch.cat((q1_s,q1_a),1)
        q1 = F.relu(self.l1(q1))
        return q1