import numpy as np
from scipy import integrate
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from matplotlib import pyplot as plt 
## INICALIZACAO DA DINAMICA DO DRONE

def eq_drone(t,x,u):
    m = 1                   #Massa em KG
    g = -10                 #Constante Gravitacional
    return np.array([x[1], (u/m+g)])
    
    
def din_drone(ti,anterior,controle):
    
    #funcao din_drone - Realimentando a dinamica da saida na entrada faz o sistema "andar pra frente" no tempo passo_t
    #esta funcao sera utilizada no processo de deep learning do drone com controles discretos (aumenta rpm do motor, ou diminui).
    #por enquanto a dinamica esta sendo simulada apenas no eixo Z, para fins de testes de viabilidade
    #a dinamica também dá um sinal de parar, e um sinal de score, se estiver dentro de uma faixa determinada de valores
    #somando 1 para cada passo de tempo dentro dos parametros, ou reset se sair fora da caixa (bounding box)
    
    b_b_min = -1          #bouding box minimo - metros
    b_b_max = 1           #bouding box maximo - metros

    
    passo_t = 0.05          #passo em segundos
    passo_w = 0.001            #passo da RPM
    B = 0.1                  #constante do motor
    
    reset = anterior[5]
    
    if reset == False:
        y_0 = anterior[2:4]
        w_0 = anterior[4]
    else:
        y_0 = np.random.rand(2)-0.5
        w_0 = 10
        score = 0
    
    #definicao da RPM a partir do sinal de controle              
    if controle ==2:
        w = w_0 + passo_w
    elif controle==1:
        w = w_0
    else:
        w = w_0 - passo_w
    
    entrada = B*w**2 

    y = (integrate.solve_ivp(eq_drone,(ti,ti+passo_t),y_0, args=[entrada])).y  #integrando a dinâmica
    
    reset = False if (y[-1,0]<b_b_max and y[-1,0]>b_b_min) else True
    score = 1-abs(y[-1,0]) if reset == False else -1
    
    return [y[0,0],y[0,1],y[-1,0],y[-1,1],w,reset,score]


## INICALIZACAO DO TREINADOR DEEP LEARNING
    
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999




class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
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
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self,name):
        self.model.save(name)


def roda_ex(roda):
    if roda== int_salva:
        plt.close('all')
    b = np.array([[0,0,0,0,0,True,0]])
    b = np.array([din_drone(0,b[0],1)])
    acao = []
    i=0
    while True:
        estado = np.array([b[i,0:5]])
        acao=np.append(acao,dqn.act_noexplore(estado))
        b = np.append(b,np.array([din_drone(0,b[i],acao[i])]),axis=0)
        if b[i+1,5]==True or i > 20/0.05:
            break
        i +=1
    plt.plot(range(len(b[:,2])),b[:,2])
    plt.draw()
    plt.pause(0.1)
    if roda%(5*int_salva)==0:
        plt.cla()

observacao = 5
acao = 3
roda = 0
int_salva = 5
media_recompensa = 0


dqn=DQNSolver(observacao,acao)
a = [0,0,0,0,0,True,0]  #inicialização da entradas, o importante é a posição 5 estar em RESET (True)
a = din_drone(0,a,1)

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
        if reset or passo > 20/0.05:
            print ("Rodada: " + str(roda) + ", Exploraçao: " + str(dqn.exploration_rate) + ", Passos: " + str(passo) + ", Pontos: " + str(int(recompensa_total)))
            break
        else:
            dqn.experience_replay()
    if roda % int_salva == 0:
        roda_ex(roda)
        str_sv = ('modelos/Modelo_'+str(roda)+'_rodadas')
        dqn.save_model(str_sv)
           