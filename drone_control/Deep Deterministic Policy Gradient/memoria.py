from collections import namedtuple, deque
import numpy as np

class salva_memoria():
    def __init__(self,n,gamma):
        self.tamanho = n
        self.memoria = deque(maxlen=self.tamanho)
        self.gamma = gamma
    def salva(self,r,estado,acao):
        self.memoria.append(np.array([r,estado,acao]))
    def calc_recompensa(self):
        if len(self.memoria) == self.tamanho:
            R = 0
            for i in self.memoria:
                R += i[0]**self.gamma
            return R,self.memoria[0][1],self.memoria[0][2]
        else:
            return 0,0,0
        
env = salva_memoria(10,0.95)

for i in range(125):
    env.salva(i,i+1,i+2)  
    R,_,_ = env.calc_recompensa()
    print(R)
print(env.memoria)