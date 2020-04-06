from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 
from collections import deque
import warnings
warnings.filterwarnings("ignore")

TR = 3 #Thrust Ratio
m, g = 1.2, 10
mod_omega = 7000
b = 1.875e-7
B = 1.875e-8
J_xx,J_yy,J_zz = 8.3e-3,8.3e-3,15.5e-3
    
d = 0.25
      
acao_neutra = np.array([[0.57143,0.57143,0.57143,0.57143]])


def mat_rot(phi,theta,psi):
    sphi = np.sin(phi)
    stheta = np.sin(theta)
    spsi = np.sin(psi)
    cphi = np.cos(phi)
    ctheta = np.cos(theta)
    cpsi = np.cos(psi)
    ttheta = np.tan(theta)
    
    R_z = np.array([[cpsi,-spsi,0],
                    [spsi,cpsi,0],
                    [0,0,1]])
    
    R_y = np.array([[ctheta,0,stheta],
                    [0,1,0],
                    [-stheta,0,ctheta]])
    
    R_x = np.array([[1,0,0],
                    [0,cphi,-sphi],
                    [0,sphi,cphi]])
    
    T = np.array([[1,sphi*ttheta,cphi*ttheta],
                    [0, cphi, -sphi],
                    [0,sphi/ctheta, cphi/ctheta]])
        
    if np.any(np.array([np.isnan(T),np.isinf(T)])):
        print('Atenção, divergencia na matriz de rotação de velocidades angulares')               
    return np.transpose(np.dot(R_z,np.dot(R_y,R_x))),T
    
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
    #                                 12   13    14   15    16   17    18     19        20     21      22       23
    # Resultado da integral em ordem: 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 phi, 7  theta, 8 psi, 9 w_xx, 10 w_yy, 11 w_zz                          
    return np.array([vel_x,accel_x,vel_y,accel_y,vel_z,accel_z,vel_phi,vel_theta,vel_psi,accel_w_xx,accel_w_yy,accel_w_zz])

    
class DinamicaDrone():

    
    def __init__(self,passo_t,n_max):
        
        plt.close('all')
        self.arrumou = False

        
        self.peso_posicao = 4
        self.peso_angulo = 0.2
        self.peso_velocidade = 0
        self.peso_velocidade_ang = 0
        self.peso_controle = 0  
        self.peso_var_controle = 0
        
        self.b_b_pos = 10
        self.b_b_point_pos = 0.5
        self.b_b_vel = 10
        self.b_b_phi = np.pi/2-0.2
        self.b_b_point_phi = np.pi/5
        self.b_b_theta = np.pi/2-0.2
        self.b_b_point_theta = np.pi/5
        self.b_b_psi = np.pi/2-0.2
        self.b_b_point_psi = np.pi/10
      
        self.estados = 12  
        
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])   

        self.n_max = n_max          
        self.passo_t = passo_t
    
        self.lista_nomes = ('x','d_x','y','d_y', 'z','d_z','phi','theta','psi','w_xx','w_yy','w_zz')      
        self.lista_acoes = ('F1','F2','F3','F4')
        
        
    def q_distancia(self,ponto,x):
        if abs(x) >= ponto:
            return 0
        else:
            return ((ponto-abs(x))/ponto)**2
    
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
        _,W = mat_rot(self.y[18],self.y[19],self.y[20])
        w_i = np.dot(W,np.array([[self.y[21]],
                                 [self.y[22]],
                                 [self.y[23]]]))
        sin_cos_entrada = np.array([np.sin(self.y[18]),np.cos(self.y[18]),np.sin(self.y[19]),np.cos(self.y[19]),np.sin(self.y[20]),np.cos(self.y[20])])
        self.entrada_agente = np.concatenate((self.y[12:18],sin_cos_entrada,np.transpose(w_i)[0],self.y[21::]))
        self.reset_fun()
        self.pontos_fun()
            
    def reset_fun(self):
        #X
        cond_x = abs(self.y[12])<self.b_b_pos
        #VX
        cond_vx = abs(self.y[13])<self.b_b_vel
        #Y
        cond_y = abs(self.y[14])<self.b_b_pos
        #VY
        cond_vy = abs(self.y[15])<self.b_b_vel
        #Z
        cond_z = abs(self.y[16])<self.b_b_pos
        #VZ
        cond_vz = abs(self.y[17])<self.b_b_vel
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
        
    def pontos_fun(self):
        
        pontos_posicao = self.q_distancia(self.b_b_point_pos,self.y[12])+self.q_distancia(self.b_b_point_pos,self.y[14])+self.q_distancia(self.b_b_point_pos,self.y[16])
        pontos_posicao *= self.peso_posicao
        pontos_angulo = self.q_distancia(self.b_b_point_phi,self.y[18])+self.q_distancia(self.b_b_point_theta,self.y[19])+self.q_distancia(self.b_b_point_psi,self.y[20])
        pontos_angulo *= self.peso_angulo
        
        if self.reset == False or not self.cond_n:
            self.pontos = pontos_posicao+pontos_angulo
        else:
            self.pontos = -200
    
        
    def impr_graf(self,estados,acoes,agent,rodada,n_rodadas):
        agent.actor_target.eval()
        agent.critic_target.eval()
        
        erros = deque(maxlen = self.n_max*n_rodadas)
        pontos = deque(maxlen = n_rodadas)
        passos = deque(maxlen = n_rodadas)
        for j in range(n_rodadas):
            self.passo(acao_neutra)
            saida_acao = acao_neutra       
            saida = np.array([self.y])
            pontos_sol = 0
            while not self.reset:
                acao = agent.act_target(self.entrada_agente,False)
                self.passo(acao)
                saida = np.append(saida,np.array([self.y]),axis=0)
                saida_acao = np.append(saida_acao,acao,axis=0)
                erros.append((abs(self.y[12])+abs(self.y[14])+abs(self.y[16]))/3)
                pontos_sol += self.pontos               
            saida_f = np.array([np.append(saida[-1,self.estados:self.estados*2],np.zeros(self.estados),axis=0)])
            saida = np.append(saida,saida_f,axis=0)
            passos.append(self.i)            
            pontos.append(pontos_sol)
        erro_media = np.mean(erros)
        ponto_media = np.mean(pontos)
        passos_media = np.mean(passos)
                        
        ax1 = plt.subplot(212)
        ax1.title.set_text('Estados')
        ax2 = plt.subplot(221)
        ax2.title.set_text('Erro')
        ax3 = plt.subplot(222)
        ax3.title.set_text('Pontos')
        box = ax1.get_position()
        if not self.arrumou:
            ax1.set_position([box.x0, box.y0, box.width * 0.90, box.height])
            self.arrumou = True
       
        ax1.cla()
        for estado in estados:
            ax1.plot(range(len(saida[:,estado])),saida[:,estado],linewidth=1,label=('Estado: '+self.lista_nomes[estado]))
        for acao in acoes:
            ax1.plot(range(len(saida_acao[:,acao])),saida_acao[:,acao],linewidth=1,linestyle=':',label=('Ação: '+self.lista_acoes[acao]))
        

        # Put a legend to the right of the current axis
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))



        ax2.scatter(rodada,erro_media,color = 'blue', marker = 'x', s = 3)              
        ax3.scatter(rodada,ponto_media,color = 'red', marker = 'x', s = 3)              

        plt.draw()
        plt.pause(0.1)
        
        print('Rodada: %i Passos %.2f Pontos: %.2f Erro: %.2f' %(rodada, passos_media,ponto_media,erro_media))

        agent.actor_target.train()
        agent.critic_target.train()
    