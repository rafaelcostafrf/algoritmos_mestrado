import numpy as np
from ddpg_agent import Agent
from collections import deque
from drone_modelo import DinamicaDrone
from torch.cuda import empty_cache
import torch

empty_cache()

rodadas = 1000


#Dinamica do sistema (passo de tempo e número máximo de iterações)
passo_t = 0.01 
n_max = 200
#inicializa o ambiente
env = DinamicaDrone(passo_t,n_max)

#imprime informaçoes a cada n rodadas
n_imp = 10

# quantidade de açoes
action_size = 4

# quantidade de estados
state_size = 18

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)
scores_deque = deque(maxlen=n_imp)
passos_deque = deque(maxlen=n_imp)
rodada = 0
# plt.close('all')

for rodada in range(rodadas):
    rodada += 1    
    env.passo(np.array([[0,0,0,0]]))
    states = env.entrada_agente
    scores = 0    
    
    while True:
        action = agent.act(states,True,rodada)
        env.passo(action)
        next_state = env.entrada_agente        
        rewards = env.pontos                     
        dones = env.reset                        
        agent.step(states, action, rewards, next_state, dones, env.i)
        states = next_state
        scores += rewards                                  
        if dones:                                 
            scores_deque.append(scores)
            passos_deque.append(env.i)
            break
    if rodada % n_imp == 0: 
        env.impr_graf([0,2,4,6,8,10],[0,1,2,3],agent,rodada)
        print('Rodada: %i Exploração: %.2f Passos %.2f Pontos: %.2f' %(rodada, agent.EXPLORATION_RATE, np.mean(passos_deque), np.mean(scores_deque)/np.mean(passos_deque)))

torch.save(agent.critic_local.state_dict(), 'modelos/critic_local.pt')
torch.save(agent.critic_target.state_dict(), 'modelos/critic_target.pt')
torch.save(agent.actor_local.state_dict(), 'modelos/actor_local.pt')
torch.save(agent.actor_target.state_dict(), 'modelos/actor_target.pt')
print('Os modelos foram salvos')