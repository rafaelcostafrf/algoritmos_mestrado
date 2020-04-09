import numpy as np
from drone_modelo import DinamicaDrone

b = 1.875e-7
a = 12/4-0.1
da = np.sqrt(a/b)/7000

c = 12/4+0.1
dc = np.sqrt(c/b)/7000

s = 12/4
ds = np.sqrt(s/b)/7000



env = DinamicaDrone(0.01, 400)

env.inicial()
env.y_0 = np.zeros([12])

acao = np.array([[da,ds,dc,ds]])
env.passo(acao)
acao = np.array([[ds,da,ds,dc]])
env.passo(acao)
acao = np.array([[ds,ds,ds,ds]])
for i in range(100):    
    env.passo(acao)
print(env.y)