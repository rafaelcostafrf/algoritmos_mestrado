import numpy as np
from scipy import integrate
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from matplotlib import pyplot as plt 

## INICIALIZACAO DOS SUPERPARAMETROS

GAMMA = 0.95
LEARNING_RATE = 0.0001

MEMORY_SIZE = int(1E+9)
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.001

observacao = 5
w_con = [-0.2,-0.05,0,0.05,0.2]
acoes = len(w_con)

n_imprime = 100
int_salva = 100
int_simula = 100
passo_max = 50
peso_contr = 0
peso_acerto = 5
acao_in = int(acoes/2)

## INICALIZACAO DA DINAMICA DO DRONE

def eq_drone(t,x,u):
    m = 1                   #Massa em KG
    g = -10                 #Constante Gravitacional
    return np.array([x[1], (u/m+g)])

def custo_controle(controle):
    custo = (abs(w_con[controle])/max(w_con))**2*peso_contr
    return custo
    
def gera_rand():
    saida = np.random.rand(2)
    saida[0] = saida[0]/2-0.25 
    saida[1] = saida[1]/10-0.05
    return saida
     
class resultados:
    
    def __init__(self,x):
        self.entrada = np.array([x])
        

    def salva(self,entrada_2):
        self.entrada = np.append(self.entrada,np.array([entrada_2]),axis=0)
        
    def imprime(self,i):
        if self.entrada[-1,0] % i == 0:
            print('Rodada: %i - %i Exploração %.2f Passos %.2f Pontos %.2f' %(self.entrada[-1-i,0],self.entrada[-1,0],np.mean(self.entrada[-1-i:-1,1]),np.mean(self.entrada[-1-i:-1,2]),np.mean(self.entrada[-1-i:-1,3])))
            plt.figure('Desenvolvimento - Erros')
            plt.scatter(self.entrada[-1,0],np.mean(self.entrada[-1-i:-1,3]), c = 'blue', marker='x', linewidths=0.1)
            plt.draw()
            plt.pause(0.1)
        if self.entrada[-1,0] % int_simula ==0:
            roda_ex()
        if self.entrada[-1,0] % int_salva == 0:
            str_sv = ('modelos/Modelo_'+str(roda)+'_rodadas')
            dqn.save_model(str_sv)
           
    
    
def din_drone(ti,anterior,controle):
    
    #funcao din_drone - Realimentando a dinamica da saida na entrada faz o sistema "andar pra frente" no tempo passo_t
    #esta funcao sera utilizada no processo de deep learning do drone com controles discretos (aumenta rpm do motor, ou diminui).
    #por enquanto a dinamica esta sendo simulada apenas no eixo Z, para fins de testes de viabilidade
    #a dinamica também dá um sinal de parar, e um sinal de score, se estiver dentro de uma faixa determinada de valores
    #somando 1 para cada passo de tempo dentro dos parametros, ou reset se sair fora da caixa (bounding box)
    
    b_b_min = -1          #bouding box minimo - metros
    b_b_max = 1           #bouding box maximo - metros

    
    passo_t = 0.1          #passo em segundos
    B = 1                  #constante do motor
    
    reset = anterior[5]    
    y_0 = anterior[2:4] if reset == False else gera_rand()
    
    #definicao da RPM a partir do sinal de controle              
    w_0 = anterior[4] if reset == False else 10
    w = w_0+w_con[controle]
    
    entrada = B*w

    y = (integrate.solve_ivp(eq_drone,(ti,ti+passo_t),y_0, args=[entrada])).y  #integrando a dinâmica
        
    reset = False if (y[0,-1]<b_b_max and y[0,-1]>b_b_min) else True
    
    if reset == False:
        # if abs(y[0,-1])<0.025:
        score = (1 - abs(y[0,-1]))**2*peso_acerto-custo_controle(controle)
        # elif abs(y[0,-1])<0.1:
        #     score = 3 - custo_controle(controle)
        # elif abs(y[0,-1])<0.25:
        #     score = 2 - custo_controle(controle)
        # else:
        #     score = 1 - custo_controle(controle)
    else:
        score = -1

    return [y[0,0],y[1,0],y[0,-1],y[1,-1],w,reset,score]


## INICALIZACAO DO TREINADOR DEEP LEARNING


    
class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(120, input_shape=(observation_space,), activation="linear"))
        self.model.add(Dense(240, activation="tanh"))
        self.model.add(Dense(240, activation="tanh"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def act_noexplore(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate = EXPLORATION_MAX*np.exp(-EXPLORATION_DECAY*roda)
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self,name):
        self.model.save(name)


def roda_ex():
    b = np.array([[0,0,0,0,0,True,0]])
    b = np.array([din_drone(0,b[0],acao_in)])
    acao = [acao_in]
    i=0
    while True:
        estado = np.array([b[i,0:5]])
        acao=np.append(acao,dqn.act_noexplore(estado))
        b = np.append(b,np.array([din_drone(0,b[i],acao[i])]),axis=0)
        if b[i+1,5]==True or i == passo_max:
            break
        i +=1
    plt.figure('Sistema - Exemplo')
    plt.cla()    
    plt.plot(range(len(b[:,2])),b[:,2])
    plt.plot(range(len(acao)),acao/acoes)
    plt.draw()
    plt.pause(0.1)


plt.close('all')
plt.ion()


roda = 0

dqn=DQNSolver(observacao,acoes)
a = [0,0,0,0,0,True,0]  #inicialização da entradas, o importante é a posição 5 estar em RESET (True)
a = din_drone(0,a,acao_in)
x = resultados([0,0,0,0])

while True:
    roda += 1
    passo = 0
    recompensa_total = 0
    while True:
        passo += 1
        estado = np.array([a[0:5]])
        acao = dqn.act(estado)
        a=din_drone(0,a,acao)
        estado_prox = np.array([a[0:5]])
        recompensa = a[6]
        reset = a[5]
        dqn.remember(estado, acao, recompensa, estado_prox, reset)
        recompensa_total += recompensa
        dqn.experience_replay()
        if reset or passo == passo_max:
            a = [0,0,0,0,0,True,0]
            a = din_drone(0,a,acao_in)
            x.salva([roda,dqn.exploration_rate,passo,recompensa_total])
            x.imprime(n_imprime)
            break