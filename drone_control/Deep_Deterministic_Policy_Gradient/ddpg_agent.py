import numpy as np
import random
from collections import namedtuple, deque
import pickle
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
BUFFER_SIZE = int(1e6)  # replay buffer size

GAMMA = 0.99            # discount factor
TAU = 0.005              # for soft update of target parameters
LR_ACTOR = 0.001      # learning rate of the actor 
LR_CRITIC = 0.001     # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
POLICY_TRAIN_FREQ = 2
BATCH_SIZE = 200       # minibatch size

MIN_MEM_SIZE = 10*BATCH_SIZE
device = torch.device("cuda:0")




class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, rodadas_totais):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.mu = 0
        self.theta = 0
        self.sigma = 5
        self.sigma_min = 2
        self.sigma_int = self.sigma
        steps_to_min = rodadas_totais/2
        self.exploration_decay = -np.log(self.sigma_min/self.sigma)/steps_to_min
        
        self.state_size = state_size
        self.action_size = action_size
        self.i = 0
        self.rodada = 0
        self.rodada_soma = 0
        # self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        try:
               self.actor_local.load_state_dict(torch.load('modelos/actor_local.pt'))
               self.actor_target.load_state_dict(torch.load('modelos/actor_target.pt'))
               self.critic_local.load_state_dict(torch.load('modelos/critic_local.pt'))
               self.critic_target.load_state_dict(torch.load('modelos/critic_target.pt'))
               print('Os modelos foram carregados')
        except:
               print('Nao foi possivel carregar os modelos salvos') 


    
    def step(self, state, action, reward, next_state, done, rodada):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.rodada = rodada
        self.memory.add(state, action, reward, next_state, done)                
        # Learn, if enough samples are available in memory
        if len(self.memory) > MIN_MEM_SIZE:
            if self.rodada_soma == 1:
                sys.stdout.write('\n')
                self.rodada_soma = 0
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        else:
            if self.rodada_soma == 0:                    
                self.rodada_soma = 1
            string = str('\rPopulando a memória: %.2f %%' %(len(self.memory)/MIN_MEM_SIZE*100))
            sys.stdout.write(string)
    
    def OU(self,x):
            self.sigma = max(self.sigma_min,self.sigma_int*np.exp(-self.exploration_decay*self.rodada))
            return self.theta * (self.mu - x) + self.sigma * np.random.randn()

        
    def act_local(self, state, add_noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
                action += np.array([[self.OU(action[0,0]),self.OU(action[0,1]),self.OU(action[0,2]),self.OU(action[0,3])]])
        return np.clip(action, 0, 9)
    
    def act_target(self, state, add_noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        if add_noise:
            action += np.array([[self.OU(action[0,0]),self.OU(action[0,1]),self.OU(action[0,2]),self.OU(action[0,3])]])
        return np.clip(action, 0, 9)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state, action, reward, next_state, done = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models       
        next_action = self.actor_target(next_state)
        with torch.no_grad():
            next_action_cte = next_action.cpu().data.numpy()
            for i in range(len(next_action_cte)):
                next_action_cte[i,:] = np.array([self.OU(next_action_cte[i,0]),self.OU(next_action_cte[i,1]),self.OU(next_action_cte[i,2]),self.OU(next_action_cte[i,3])])
            next_action_noise = torch.FloatTensor(next_action_cte).to(device)
        next_action += next_action_noise
        next_action=next_action.clamp(0,9)

        Q1_alvo, Q2_alvo = self.critic_target(next_state,next_action)
        
        target_Q = reward+((1-done)*GAMMA*torch.min(Q1_alvo,Q2_alvo)).detach()
        
        Q1_at, Q2_at = self.critic_local(state,action)
        
        self.critic_loss = F.mse_loss(Q1_at,target_Q)+F.mse_loss(Q2_at,target_Q)
        
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        self.i += 1
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        if self.i % POLICY_TRAIN_FREQ == 0:
            self.actor_loss = -self.critic_local.Q1(state, self.actor_local(state)).mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()   
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

        