from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 

TR = 3 #Thrust Ratio
m, g = 1.2, 10
mod_omega = 3500


def eq_drone(t,x,R,entrada):
    
    J_phi,J_theta,J_psi = 8.3e-3,8.3e-3,15.5e-3
    
    d = 0.25
    b = 1.875e-7
    B = 1.875e-8
    
    w1 = (entrada[0,0]+1)*mod_omega
    w2 = (entrada[0,1]+1)*mod_omega
    w3 = (entrada[0,2]+1)*mod_omega
    w4 = (entrada[0,3]+1)*mod_omega
    
    f1 = b*w1**2
    f2 = b*w2**2
    f3 = b*w3**2
    f4 = b*w4**2
    
    m1 = B*w1**2
    m2 = B*w2**2
    m3 = B*w3**2
    m4 = B*w4**2

    F = np.array([[0],[0],[(f1+f2+f3+f4)]])
    F_VEC = np.dot(R,F)

    vel_x = x[1]
    accel_x = F_VEC[0]/m

    vel_y = x[3]
    accel_y = F_VEC[1]/m

    vel_z = x[5]
    accel_z = F_VEC[2]/m-g 

    
    vel_phi = x[7]
    accel_phi = (f1-f3)*d/J_phi
    
    
    vel_theta = x[9]
    accel_theta = (f2-f4)*d/J_theta
    
    
    vel_psi = x[11]
    accel_psi = (m1-m2+m3-m4)/J_psi
                              
    return np.array([vel_x,accel_x,vel_y,accel_y,vel_z,accel_z,vel_phi,accel_phi,vel_theta,accel_theta,vel_psi,accel_psi])

class DinamicaDrone():
    
    def __init__(self,passo_t,n_max):
        self.peso_posicao = 3
        self.peso_angulo = 0
        self.peso_velocidade = 1 
        self.peso_velocidade_ang = 0
        self.peso_controle = 0       
        
        self.b_b_z = 1
        self.b_b_vz = 3
        self.b_b_phi = 0.75
        self.b_b_theta = 0.75
        self.b_b_psi = 0.75
      
        self.estados = 12  
        
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])   

        self.n_max = n_max          
        self.passo_t = passo_t
    
        self.lista_nomes = ('x','d_x','y','d_y', 'z','d_z','phi','d_phi','theta','d_theta','psi','d_psi')      
        self.lista_acoes = ('F1','F2','F3','F4')
    def mat_rot(self,phi,theta,psi):
        sphi = np.sin(phi)
        stheta = np.sin(theta)
        spsi = np.sin(psi)
        cphi = np.cos(phi)
        ctheta = np.cos(theta)
        cpsi = np.cos(psi)
        
        R_z = np.array([[cpsi,spsi,0],
                        [-spsi,cpsi,0],
                        [0,0,1]])
        
        R_y = np.array([[ctheta,0,-stheta],
                        [0,1,0],
                        [stheta,0,ctheta]])
        
        R_x = np.array([[1,0,0],
                        [0,cphi,sphi],
                        [0,-sphi,cphi]])
                       
        return np.dot(R_x,np.dot(R_y,R_z))             
        
        
    def passo(self,acao):
        
        if self.reset == False:
            y_0 = self.y[self.estados:2*self.estados+1]
            self.i += 1
        else:
            y_0 = np.random.rand(self.estados)-0.5
            self.reset = False
            self.i = 0
            self.mat_rot(y_0[6],y_0[8],y_0[10])
        
        self.entrada = acao       
        
        
        self.saida = (integrate.solve_ivp(eq_drone,(0,self.passo_t),y_0, args=[self.R,self.entrada])).y
        self.R = self.mat_rot(self.y[18],self.y[20],self.y[22])
        
        self.y = np.append(np.transpose(self.saida[:,0]),np.transpose(self.saida[:,-1]))
        self.entrada_agente = np.concatenate((self.y[12:18],np.array([self.y[19],self.y[21],self.y[23]]),self.R[0,:],self.R[1,:],self.R[2,:]),axis=0)
        self.reset_fun()
        self.pontos_fun(0)
        self.estado_novo = self.y[self.estados:2*self.estados+1]
    
    def passo_volta(self,acao,i):
        entrada = acao
        y_0 = self.y[self.estados:2*self.estados+1] 
        saida = (integrate.solve_ivp(eq_drone,(0,self.passo_t*i),y_0, args=[self.R,entrada])).y
        y = np.append(np.transpose(saida[:,0]),np.transpose(saida[:,-1]))
        R = self.mat_rot(y[18],y[20],y[22])
        entrada_agente = np.concatenate((y[12:18],np.array([y[19],y[21],y[23]]),R[0,:],R[1,:],R[2,:]),axis=0)
        return entrada_agente
        
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
        cond_theta = abs(self.y[20])<self.b_b_theta
        #PSI
        cond_psi = abs(self.y[22])<self.b_b_psi
        #N
        self.cond_n = self.i < self.n_max
        
        if all([cond_x,cond_vx,cond_y,cond_vy,cond_z,cond_vz,cond_phi,cond_theta,cond_psi,self.cond_n]):
            self.reset = False 
        else: 
            self.reset = True
        
    def pontos_fun(self,u):
        
        pontos_posicao = (np.linalg.norm([1,1,1])-np.linalg.norm(np.array([self.y[12],self.y[14],self.y[16]])))*self.peso_posicao
        
        
        pontos_velocidade_pos = (np.linalg.norm([2,2,2])-np.linalg.norm(np.array([self.y[13],self.y[15],self.y[17]])))*self.peso_velocidade

        
        if self.reset == False or not self.cond_n:
            self.pontos = pontos_posicao+pontos_velocidade_pos+1
        else:
            self.pontos = -10
    
        
    def impr_graf(self,estados,acoes,agent,rodada):
        self.passo(np.array([[0,0,0,0]]))
        saida_acao = np.array([[0,0,0,0]])        
        saida = np.array([self.y])

        while not self.reset:
            acao = agent.act(self.entrada_agente,False,rodada)
            self.passo(acao)
            saida = np.append(saida,np.array([self.y]),axis=0)
            saida_acao = np.append(saida_acao,acao,axis=0)
        saida_f = np.array([np.append(saida[-1,self.estados:self.estados*2],np.zeros(self.estados),axis=0)])
        erro = 0
        for i in saida_f:
            erro += np.linalg.norm(i)/self.i
        
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