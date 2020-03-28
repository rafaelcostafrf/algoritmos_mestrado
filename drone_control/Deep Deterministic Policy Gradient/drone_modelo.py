from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 


def eq_drone(t,x,u):
    m, g = 1, -10                               
    return np.array([x[1], (u/m+g)])

class DinamicaDrone():
    
    def __init__(self):
        self.peso_acerto = 2
        self.peso_controle = 0.5
        self.b_b_min = -1          
        self.b_b_max = 1           
        self.passo_t = 0.1          
        self.B = 1
        self.estados = 2  
        self.observacao = 5
        self.n_max = 200
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])             
        self.entra_rn = np.zeros([self.observacao])
        
    def passo(self,u):
        if self.reset == False:
            y_0 = self.y[self.estados:2*self.estados+1]
            self.i += 1
        else:
            y_0 = np.random.rand(2)-0.5
            self.reset = False
            self.i = 0
        entrada = u + 10
        self.saida = (integrate.solve_ivp(eq_drone,(0,self.passo_t),y_0, args=[entrada])).y
        self.y = np.append(np.transpose(self.saida[:,0]),np.transpose(self.saida[:,-1]))
        self.reset_fun()
        self.pontos_fun(u)
    
    def reset_fun(self):  
        if (self.y[2]<self.b_b_max and self.y[2]>self.b_b_min) and self.i < self.n_max: 
            self.reset = False 
        else: 
            self.reset = True
        
    def pontos_fun(self,u):
        if self.reset == False:
            self.pontos = (1 - abs(self.y[2]))**2*self.peso_acerto-u**2*self.peso_controle
        else:
            self.pontos = -1
        
    def impr_graf(self,estados,agent):
        self.passo(0)
        saida_acao = np.array([0])        
        saida = np.array([self.y])
        while not self.reset:
            acao = agent.act(self.y,False,0)
            self.passo(acao)
            saida = np.append(saida,np.array([self.y]),axis=0)
            saida_acao = np.append(saida_acao,np.array([acao]))
        saida_f = np.array([np.append(saida[-1,self.estados:self.estados*2],np.zeros(self.estados),axis=0)])
        saida = np.append(saida,saida_f,axis=0)
        plt.figure('Estados')
        plt.cla()
        for estado in estados:
            plt.plot(range(len(saida[:,estado])),saida[:,estado],linewidth=1)
        plt.plot(range(len(saida_acao)),saida_acao,linewidth=1)
        plt.pause(0.01)