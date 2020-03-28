import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from collections import deque
from drone_modelo import DinamicaDrone

#imprime informaçoes
n_imp = 30

#inicializa o ambiente
env = DinamicaDrone()

# size of each action
action_size = 1

# examine the state space
state_size = 4

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)
scores_deque = deque(maxlen=n_imp)
passos_deque = deque(maxlen=n_imp)
rodada = 0
plt.close('all')
while True:
    rodada += 1    
    env.passo(10)
    states = env.y
    scores = 0    
    
    while True:
        action = agent.act(states,True,rodada)
        env.passo(action)
        next_state = env.y         
        rewards = env.pontos                     
        dones = env.reset                        
        agent.step(states, action, rewards, next_state, dones)
        states = next_state
        scores += rewards                                  
        if dones:                                 
            scores_deque.append(scores)
            passos_deque.append(env.i)
            break
    if rodada % n_imp == 0: 
        env.impr_graf([0],agent)
        plt.figure('Evolução - Score')
        plt.scatter(rodada,np.mean(scores_deque),color = 'blue', marker = 'x', s = 3)
        plt.pause(0.01)
        print('Rodada: %i Exploração: %.2f Passos %.2f Pontos: %.2f' %(rodada, agent.EXPLORATION_RATE, np.mean(passos_deque), np.mean(scores_deque)))


