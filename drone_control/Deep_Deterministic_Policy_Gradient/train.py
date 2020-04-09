from ddpg_agent import Agent
from collections import deque
from drone_modelo import DinamicaDrone
from torch.cuda import empty_cache
import torch
import sys

empty_cache()

rodadas = 1000

#Dinamica do sistema (passo de tempo e número máximo de iterações)
passo_t = 0.01 
n_max = 600
n_max_sim = 1000
#inicializa o ambiente
env = DinamicaDrone(passo_t,n_max)
env_sim = DinamicaDrone(passo_t,n_max_sim)

#imprime informaçoes a cada n rodadass
n_imp = 10

# quantidade de açoes
action_size = 4

# quantidade de estados
state_size = 18

agent = Agent(state_size, action_size, 1, rodadas)
scores_deque = deque(maxlen=n_imp)
passos_deque = deque(maxlen=n_imp)
perca_ator = deque(maxlen=n_max)
perca_critico = deque(maxlen=n_max) 
rodada = 0
# plt.close('all')


while rodada < rodadas:
    rodada += 1
    scores = 0   
    env.inicial()
    states = env.entrada_agente  
    while True:
        action = agent.act_local(states,True)
        env.passo(action)
        next_state = env.entrada_agente  
        rewards = env.pontos
        scores += rewards
        dones = env.reset
        agent.step(states, action, rewards, next_state, dones, rodada)
        states = next_state
        if dones:
            if rodada>0:
                scores_deque.append(scores)
                passos_deque.append(env.i)
            break
        if agent.rodada_soma == 0:
            string = str('\rTreinando %.2f %%' %(((rodada-1)%n_imp+env.i/n_max)/n_imp*100))
            sys.stdout.write(string)                                     
    rodada -= agent.rodada_soma
    if rodada % n_imp == 0 and not agent.rodada_soma:
        env_sim.impr_graf([0,2,4,6,7,8],[0,1,2,3],agent,rodada,rodadas,5)
        torch.save(agent.critic_local.state_dict(), 'modelos/critic_local_checkpoint.pt')
        torch.save(agent.critic_target.state_dict(), 'modelos/critic_target_checkpoint.pt')
        torch.save(agent.actor_local.state_dict(), 'modelos/actor_local_checkpoint.pt')
        torch.save(agent.actor_target.state_dict(), 'modelos/actor_target_checkpoint.pt')
        


torch.save(agent.critic_local.state_dict(), 'modelos/critic_local.pt')
torch.save(agent.critic_target.state_dict(), 'modelos/critic_target.pt')
torch.save(agent.actor_local.state_dict(), 'modelos/actor_local.pt')
torch.save(agent.actor_target.state_dict(), 'modelos/actor_target.pt')
print('Os modelos foram salvos')
