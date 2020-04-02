import numpy as np
from ddpg_agent import Agent
from collections import deque
from drone_modelo import DinamicaDrone
from torch.cuda import empty_cache
import matplotlib.pyplot as plt
import torch
import warnings


empty_cache()

rodadas = 3000
acao_neutra = np.array([[0.57143,0.57143,0.57143,0.57143]])

#Dinamica do sistema (passo de tempo e número máximo de iterações)
passo_t = 0.01 
n_max = 500
#inicializa o ambiente
env = DinamicaDrone(passo_t,n_max)


#imprime informaçoes a cada n rodadass
n_imp = 100

# quantidade de açoes
action_size = 4

# quantidade de estados
state_size = 18

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)
scores_deque = deque(maxlen=n_imp)
passos_deque = deque(maxlen=n_imp)
perca_ator = deque(maxlen=n_max)
perca_critico = deque(maxlen=n_max) 
rodada = 0
# plt.close('all')


for k in range(rodadas):
    env.reset=True
    rodada += 1
    env.passo(acao_neutra)
    states = env.entrada_agente
    scores = 0    
    
    while True:
        action = agent.act(states,True,rodada)
        env.passo(action)
        next_state = env.entrada_agente        
        rewards = env.pontos
        scores += rewards
        dones = env.reset
        agent.step(states, action, rewards, next_state, dones, env.i)
        try:
            perca_ator.append(agent.actor_loss.data)
            perca_critico.append(agent.critic_loss.data)
        except:
            pass
        if dones:
            m_perca_ator = np.mean(perca_ator)
            perca_ator.clear()
            m_perca_critico = np.mean(perca_critico)
            perca_critico.clear()
            scores_deque.append(scores)
            passos_deque.append(env.i)
            break
        states = next_state                                
    rodada -= agent.rodada_soma
    if rodada % n_imp == 0 and not agent.rodada_soma: 
        env.impr_graf([0,2,4,6,8,10],[0,1,2,3],agent,rodada)
        print('Rodada: %i Passos %.2f Perca Critico: %.2f Perca Ator: %.2f Pontos: %.2f' %(rodada, np.mean(passos_deque), m_perca_critico, m_perca_ator, np.mean(scores_deque)))

torch.save(agent.critic_local.state_dict(), 'modelos/critic_local.pt')
torch.save(agent.critic_target.state_dict(), 'modelos/critic_target.pt')
torch.save(agent.actor_local.state_dict(), 'modelos/actor_local.pt')
torch.save(agent.actor_target.state_dict(), 'modelos/actor_target.pt')
print('Os modelos foram salvos')
