from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 
from collections import deque

TR = 3 #Thrust Ratio
m, g = 1.2, 10
mod_omega = 7000
b = 1.875e-7
B = 1.875e-8
J_xx,J_yy,J_zz = 8.3e-3,8.3e-3,15.5e-3
    
d = 0.25
      
acao_neutra = np.array([[0.57143,0.57143,0.57143,0.57143]])
print(acao_neutra)

def mat_rot(phi,theta,psi):
    sphi = np.sin(phi)
    stheta = np.sin(theta)
    spsi = np.sin(psi)
    cphi = np.cos(phi)
    ctheta = np.cos(theta)
    cpsi = np.cos(psi)
    
    R_z = np.array([[cpsi,-spsi,0],
                    [spsi,cpsi,0],
                    [0,0,1]])
    
    R_y = np.array([[ctheta,0,stheta],
                    [0,1,0],
                    [-stheta,0,ctheta]])
    
    R_x = np.array([[1,0,0],
                    [0,cphi,-sphi],
                    [0,sphi,cphi]])
    
    W_R = np.array([[1,0,-stheta],
                    [0, cphi, ctheta*sphi],
                    [0,-sphi, ctheta*cphi]])
    W_R_Inv = np.linalg.inv(W_R)
    
    if np.any(np.array([np.isnan(W_R_Inv),np.isinf(W_R_Inv)])):
        print('Atenção, divergencia na matriz de rotação de velocidades angulares')               
    return np.transpose(np.dot(R_z,np.dot(R_y,R_z))),W_R_Inv
    
def eq_drone(t,x,entrada):    

    w1 = (entrada[0,0])*mod_omega #Velocidade angulares dos motores
    w2 = (entrada[0,1])*mod_omega
    w3 = (entrada[0,2])*mod_omega
    w4 = (entrada[0,3])*mod_omega
    
    f1 = b*(w1**2) #Forças e momentos nos motores
    f2 = b*(w2**2)
    f3 = b*(w3**2)
    f4 = b*(w4**2)
    
    m1 = B*(w1**2)
    m2 = B*(w2**2)
    m3 = B*(w3**2)
    m4 = B*(w4**2)
         
    F = np.array([[0],[0],[(f1+f2+f3+f4)]])
      
    pos_phi = x[6]
    pos_theta = x[7]
    pos_psi = x[8]
    
    R,W_R = mat_rot(pos_phi,pos_theta,pos_psi)
    F_VEC = np.dot(R,F)

    accel_w_xx = (f1-f3)*d/J_xx
    accel_w_yy = (f2-f4)*d/J_yy
    accel_w_zz = (m1-m2+m3-m4)/J_zz
    
    W = np.array([[x[9]],
                  [x[10]],
                  [x[11]]])
    
    V_eul = np.dot(W_R,W)

    vel_phi=V_eul[0]
    vel_theta=V_eul[1]
    vel_psi=V_eul[2]
    
    vel_x = x[1]
    accel_x = F_VEC[0]/m

    vel_y = x[3]
    accel_y = F_VEC[1]/m

    vel_z = x[5]
    accel_z = F_VEC[2]/m-g 
    
    # Resultado da integral em ordem: 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 phi, 7  theta, 8 psi, 9 w_xx, 10 w_yy, 11 w_zz                          
    return np.array([vel_x,accel_x,vel_y,accel_y,vel_z,accel_z,vel_phi,vel_theta,vel_psi,accel_w_xx,accel_w_yy,accel_w_zz])

    
class DinamicaDrone():
    
    def __init__(self,passo_t,n_max):
        self.peso_posicao = 2
        self.peso_angulos = 0.2
        self.peso_velocidade = 0
        self.peso_velocidade_ang = 0
        self.peso_controle = 0  
        self.peso_var_controle = 0
        
        self.b_b_z = 3
        self.b_b_vz = 3
        self.b_b_phi = np.pi/2-0.1
        self.b_b_theta = np.pi/2-0.1
        self.b_b_psi = np.pi/2-0.1
      
        self.estados = 12  
        
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])   

        self.n_max = n_max          
        self.passo_t = passo_t
    
        self.lista_nomes = ('x','d_x','y','d_y', 'z','d_z','phi','theta','psi','w_xx','w_yy','w_zz')      
        self.lista_acoes = ('F1','F2','F3','F4')
        
        
    def passo(self,acao):
        
        if self.reset == False:
            y_0 = self.y[self.estados:2*self.estados+1]
            self.i += 1
        else:
            y_0 = np.random.rand(self.estados)-0.5
            self.acao_ant = acao_neutra
            self.reset = False
            self.i = 0           
        self.entrada = acao     
                
        self.saida = (integrate.solve_ivp(eq_drone,(0,self.passo_t),y_0, args=[self.entrada]))

        self.saida = self.saida.y
        self.y = np.append(np.transpose(self.saida[:,0]),np.transpose(self.saida[:,-1]))        
        self.entrada_agente = self.y[12:25]
        self.reset_fun()
        self.pontos_fun(self.acao_ant,acao)
            
    def reset_fun(self):
        #X
        cond_x = abs(self.y[12])<self.b_b_z
        #VX
        cond_vx = abs(self.y[13])<self.b_b_vz
        #Y
        cond_y = abs(self.y[14])<self.b_b_z
        #VY
        cond_vy = abs(self.y[15])<self.b_b_vz
        #Z
        cond_z = abs(self.y[16])<self.b_b_z
        #VZ
        cond_vz = abs(self.y[17])<self.b_b_vz
        #PHI
        cond_phi = abs(self.y[18])<self.b_b_phi
        #THETA
        cond_theta = abs(self.y[19])<self.b_b_theta
        #PSI
        cond_psi = abs(self.y[20])<self.b_b_psi
        #N
        self.cond_n = self.i < self.n_max
        
        if all([cond_x,cond_vx,cond_y,cond_vy,cond_z,cond_vz,cond_phi,cond_theta,cond_psi,self.cond_n]):
            self.reset = False 
        else:
            self.reset = True
        
    def pontos_fun(self,acao_ant,acao):
        
        pontos_posicao = ((np.linalg.norm([10,10,10])-np.linalg.norm(np.array([self.y[12],self.y[14],self.y[16]])))/10)**4*self.peso_posicao
        
        pontos_angulo = ((np.linalg.norm([np.pi,np.pi,np.pi])-np.linalg.norm(np.array([self.y[18],self.y[19],self.y[20]])))/np.pi)**4*self.peso_angulos
        
        pontos_velocidade_pos = ((np.linalg.norm([3,3,3])-np.linalg.norm(np.array([self.y[13],self.y[15],self.y[17]])))/3)**4*self.peso_velocidade

        pontos_controle = -((np.linalg.norm(acao_neutra-acao))**4*self.peso_controle)
        
        pontos_var_controle = -((np.linalg.norm(acao_ant-acao))**4*self.peso_var_controle)
        
        if self.reset == False or not self.cond_n:
            self.pontos = pontos_posicao+pontos_velocidade_pos+pontos_angulo+pontos_controle+pontos_var_controle
        else:
            self.pontos = -100
    
        
    def impr_graf(self,estados,acoes,agent,rodada):
        self.reset = True
        self.passo(acao_neutra)
        saida_acao = acao_neutra       
        saida = np.array([self.y])
        while not self.reset:
            acao = agent.act_target(self.entrada_agente)
            self.passo(acao)
            saida = np.append(saida,np.array([self.y]),axis=0)
            saida_acao = np.append(saida_acao,acao,axis=0)
        saida_f = np.array([np.append(saida[-1,self.estados:self.estados*2],np.zeros(self.estados),axis=0)])
        erro = 0
        for i in saida_f:
            erro += np.linalg.norm(i)
        erro *= 1/self.i
        plt.figure('Evolução - perca')
        plt.scatter(rodada,erro,color = 'blue', marker = 'x', s = 3)  
        plt.yscale('log')             
        plt.draw()
        plt.pause(0.1)
        
        saida = np.append(saida,saida_f,axis=0)
        plt.figure('Estados')
        plt.cla()
        for estado in estados:
            plt.plot(range(len(saida[:,estado])),saida[:,estado],linewidth=1,label=('Estado: '+self.lista_nomes[estado]))
        for acao in acoes:
            plt.plot(range(len(saida_acao[:,acao])),saida_acao[:,acao],linewidth=1,linestyle=':',label=('Ação: '+self.lista_acoes[acao]))
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        
class salva_memoria():
    def __init__(self,n,gamma):
        self.tamanho = n+1
        self.memoria = deque(maxlen=self.tamanho)
        self.gamma = gamma
    def salva(self,estado,acao,r,proximo_estado,reset):
        self.memoria.append([estado,acao,r,proximo_estado,reset])
    def calc_recompensa(self):
        R = 0
        for i,j in zip(self.memoria,range(len(self.memoria))):
            R += i[2]*(self.gamma**j)
            
        return self.memoria[0][0],self.memoria[0][1],self.memoria[0][2],self.memoria[0][3],self.memoria[0][4]
    def limpa(self):
        self.memoria.clear()