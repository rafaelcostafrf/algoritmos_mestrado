from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 


def eq_drone(t,x,phi,theta,entrada):
    
    m, g = 0.35, -10
    J_phi,J_theta,J_psi = 0.06,0.06,0.1131
    B = 0.1
    b = 0.12

    f1 = entrada[0,0]*2+m/4
    f2 = entrada[0,1]*2+m/4
    f3 = entrada[0,2]*2+m/4
    f4 = entrada[0,3]*2+m/4
    
    vel_z = x[1]
    accel_z = (f1+f2+f3+f4)*(np.cos(theta)*np.cos(phi))/m+g 
    
    
    vel_phi = x[3]
    accel_phi = (f3-f4)*b/J_phi
    
    
    vel_theta = x[5]
    accel_theta = (f2-f1)*b/J_theta
    
    
    vel_psi = x[7]
    accel_psi = (f4+f3-f1-f2)*B/J_psi
                              
    return np.array([vel_z,accel_z,vel_phi,accel_phi,vel_theta,accel_theta,vel_psi,accel_psi])

class DinamicaDrone():
    
    def __init__(self,passo_t,n_max):
        self.peso_acerto = 2
        self.peso_controle = 0.2
                
        self.b_b_z = 1
        self.b_b_phi = 0.785398
        self.b_b_theta = 0.785398
        self.b_b_psi = 0.785398
          
      
        self.estados = 8  
        
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])   

        self.n_max = n_max          
        self.passo_t = passo_t
    
        self.lista_nomes = ('z','d_z','phi','d_phi','theta','d_theta','psi','d_psi')      
        self.lista_acoes = ('F1','F2','F3','F4')
    
    def passo(self,acao):
        
        if self.reset == False:
            y_0 = self.y[self.estados:2*self.estados+1]
            self.i += 1
        else:
            y_0 = np.random.rand(self.estados)-0.5
            self.reset = False
            self.i = 0
        self.entrada = acao
        phi = self.y[10]
        theta = self.y[12]
        
        self.saida = (integrate.solve_ivp(eq_drone,(0,self.passo_t),y_0, args=[phi,theta,self.entrada])).y
        self.y = np.append(np.transpose(self.saida[:,0]),np.transpose(self.saida[:,-1]))
        self.reset_fun()
        self.pontos_fun(0)
        self.estado_novo = self.y[self.estados:2*self.estados+1]
    
    def reset_fun(self):
        #Z
        cond_z = abs(self.y[8])<self.b_b_z
        #PHI
        cond_phi = abs(self.y[10])<self.b_b_phi
        #THETA
        cond_theta = abs(self.y[12])<self.b_b_theta
        #PSI
        cond_psi = abs(self.y[14])<self.b_b_psi
        #N
        cond_n = self.i < self.n_max
        
        if all([cond_z,cond_phi,cond_theta,cond_psi,cond_n]):
            self.reset = False 
        else: 
            self.reset = True
        
    def pontos_fun(self,u):
        #Z
        pontos_z = (1-abs(self.y[8]))**2*self.peso_acerto
        #PHI
        pontos_phi = (1-abs(self.y[10])/0.785398)**2*self.peso_acerto
        #THETA
        pontos_theta = (1-abs(self.y[12])/0.785398)**2*self.peso_acerto
        #PSI
        pontos_psi = (1-abs(self.y[14])/0.785398)**2*self.peso_acerto
        #N
        pontos_n = 0
        #CUSTO ENTRADA
        custo_entrada=0
        for j in self.entrada[0]:
            custo_entrada = custo_entrada+(1-abs(j))**2*self.peso_controle
        
        if self.reset == False:
            self.pontos = pontos_z+pontos_phi+pontos_theta+pontos_psi+pontos_n+custo_entrada
        else:
            self.pontos = -1
        
    def impr_graf(self,estados,acoes,agent,rodada):
        self.passo(np.array([[0,0,0,0]]))
        saida_acao = np.array([[0,0,0,0]])        
        saida = np.array([self.y])
        while not self.reset:
            acao = agent.act(self.y,True,rodada)
            self.passo(acao)
            saida = np.append(saida,np.array([self.y]),axis=0)
            saida_acao = np.append(saida_acao,acao,axis=0)
        saida_f = np.array([np.append(saida[-1,self.estados:self.estados*2],np.zeros(self.estados),axis=0)])
        saida = np.append(saida,saida_f,axis=0)
        plt.figure('Estados')
        plt.cla()
        for estado in estados:
            plt.plot(range(len(saida[:,estado])),saida[:,estado],linewidth=1,label=('Estado: '+self.lista_nomes[estado]))
        for acao in acoes:
            plt.plot(range(len(saida_acao[:,acao])),saida_acao[:,acao],linewidth=1,linestyle=':',label=('Ação: '+self.lista_acoes[acao]))
        plt.legend()
        plt.pause(0.01)