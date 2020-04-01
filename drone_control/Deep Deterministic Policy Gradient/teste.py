# Testes da dinamica 
from drone_modelo import DinamicaDrone
import numpy as np

#Dinamica do sistema (passo de tempo e número máximo de iterações)
passo_t = 0.01 
n_max = 600
#inicializa o ambiente
env = DinamicaDrone(passo_t,n_max)


entrada = np.array([[1/3*2-1+0.1,1/3*2-1,1/3*2-1-0.1,1/3*2-1]])

env.passo(entrada)
env.y = np.zeros([24])
while not env.reset:
    env.passo(entrada)
print(env.y)