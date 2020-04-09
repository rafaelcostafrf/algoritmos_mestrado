from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 
from collections import deque
import warnings
import torch
warnings.filterwarnings("ignore")
import time
import inspect
import sys
TR = 3 #Thrust Ratio
m, g = 1.2, 10
mod_omega = 1
b = 1.875e-7
B = 1.875e-8
J_xx,J_yy,J_zz = 8.3e-3,8.3e-3,15.5e-3
    
d = 0.25

pos = {'x': 0,
       'vx': 1,
       'y': 2,
       'vy': 3,
       'z': 4,
       'vz': 5,
       'phi': 6,
       'theta': 7,
       'psi': 8,
       'w_xx': 9,
       'w_yy': 10,
       'w_zz': 11}

    
class DinamicaDrone():

    
    def __init__(self,passo_t,n_max):
        
        plt.close('all')
        self.arrumou = False
        plt.figure()
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        self.peso_posicao = 2
        self.peso_angulo = 0.5
        self.peso_velocidade = 1
        self.peso_controle = 0.1  

        
        self.b_b_pos = 1
        self.b_b_vel = 1000
        self.b_b_phi = np.pi/2-0.1
        self.b_b_theta = np.pi/2-0.1
        self.b_b_psi = np.pi/2-0.1
        
        self.pontos_max = 3*self.peso_posicao*(np.sqrt(self.b_b_pos))+3*self.peso_velocidade*(np.sqrt(self.b_b_vel))+3*self.peso_angulo*(np.sqrt(self.b_b_phi))+4*self.peso_controle*(9**2)

        
        self.estados = 12  
        
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])   

        self.n_max = n_max          
        self.passo_t = passo_t
    
        self.lista_nomes = ('x','d_x','y','d_y', 'z','d_z','phi','theta','psi','w_xx','w_yy','w_zz')      
        self.lista_acoes = ('F1','F2','F3','F4')
        
        
    def mat_rot(self,phi,theta,psi):
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
            stack = inspect.stack()
            the_class = stack[1][0].f_locals["self"].__class__.__name__
            the_method = stack[1][0].f_code.co_name
            print('ATENCAO: DIVERGENCIA NA MATRIZ DE VELOCIDADES')
            print("Fui chamado pela {}.{}()".format(the_class, the_method))
            print(self.reset,self.entrada,self.entrada_agente)
            time.sleep(1)
        return np.transpose(np.dot(R_z,np.dot(R_y,R_x))),T
        
    def eq_drone(self,t,x,entrada):    
        f1 = entrada[0,0]
        f2 = entrada[0,1]
        f3 = entrada[0,2]
        f4 = entrada[0,3]
        
        m1 = 0.1*f1
        m2 = 0.1*f2
        m3 = 0.1*f3
        m4 = 0.1*f4
     
        F = np.array([[0],[0],[(f1+f2+f3+f4)]])
          
        pos_phi = x[6]
        pos_theta = x[7]
        pos_psi = x[8]
        
        R,W_R = self.mat_rot(pos_phi,pos_theta,pos_psi)
        F_VEC = np.dot(R,F)
    
        accel_w_xx = (f1-f3)*d/J_xx
        accel_w_yy = (f2-f4)*d/J_yy
        accel_w_zz = (m1+m3-m2-m4)/J_zz
        
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
            
    def q_distancia(self,ponto,x):
        if abs(x) >= ponto:
            return 0
        else:
            return ((ponto-abs(x))/ponto)**2
    
    def inicial(self):
        del self.y
        self.y_0 = np.random.rand(self.estados)-0.5
        self.reset = False
        self.i = 0
        
        _,W = self.mat_rot(self.y_0[6],self.y_0[7],self.y_0[8])
        W = np.dot(W,np.transpose(self.y_0[9:12]))
        
        
        sin_cos_entrada = np.array([np.sin(self.y_0[6]),np.cos(self.y_0[6]),np.sin(self.y_0[7]),np.cos(self.y_0[7]),np.sin(self.y_0[8]),np.cos(self.y_0[8])])
        
        self.entrada_agente = np.concatenate((self.y_0[:6:],sin_cos_entrada,self.y_0[9:12],np.transpose(W)))
        self.entrada_agente = np.array([self.entrada_agente])
    
    def passo(self,acao):            
        self.i += 1        
        self.entrada = acao                   
        self.saida = (integrate.solve_ivp(self.eq_drone,(0,self.passo_t),self.y_0, args=[self.entrada]))
        
        self.saida = self.saida.y
        self.y = np.transpose(self.saida[:,-1])
        
        
        R,W = self.mat_rot(self.y[6],self.y[7],self.y[8])
        W = np.dot(W,np.transpose(self.y[9:12]))
        
        sin_cos_entrada = np.array([np.sin(self.y[6]),np.cos(self.y[6]),np.sin(self.y[7]),np.cos(self.y[7]),np.sin(self.y[8]),np.cos(self.y[8])])
        
        self.entrada_agente = np.concatenate((self.y_0[:6:],sin_cos_entrada,self.y_0[9:12],np.transpose(W)))
        
        
        self.entrada_agente = np.concatenate((self.y[:6:],sin_cos_entrada,self.y[9:12],np.transpose(W)))
        self.entrada_agente = np.array([self.entrada_agente])
        self.reset_fun()
        self.pontos_fun()
        self.y_0 = self.y
        
    def reset_fun(self):
        #X
        cond_x = abs(self.y[0])<self.b_b_pos
        #VX
        cond_vx = abs(self.y[1])<self.b_b_vel
        #Y
        cond_y = abs(self.y[2])<self.b_b_pos
        #VY
        cond_vy = abs(self.y[3])<self.b_b_vel
        #Z
        cond_z = abs(self.y[4])<self.b_b_pos
        #VZ
        cond_vz = abs(self.y[5])<self.b_b_vel
        #PHI
        cond_phi = abs(self.y[6])<self.b_b_phi
        #THETA
        cond_theta = abs(self.y[7])<self.b_b_theta
        #PSI
        cond_psi = abs(self.y[8])<self.b_b_psi
        #N
        self.cond_n = self.i < self.n_max
        
        if all([cond_x,cond_vx,cond_y,cond_vy,cond_z,cond_vz,cond_phi,cond_theta,cond_psi,self.cond_n]):
            self.reset = False 
        else:
            self.reset = True
        
    def pontos_fun(self):
        self.pontos = self.pontos_max
        for x,vx,ang in zip(self.y[0:5:2],self.y[1:6:2],self.y[6:9]):
            self.pontos += -self.peso_posicao*(np.sqrt(abs(x)))-self.peso_velocidade*(np.sqrt(abs(vx)))-self.peso_angulo*(np.sqrt(abs(ang)))
            if abs(x) < 0.05 and abs(vx) < 0.05:
                self.pontos += self.pontos_max
        for a in self.entrada[0]:
            self.pontos += -self.peso_controle*(a**2)
        self.pontos *= 1/4/self.pontos_max
        if self.reset and self.i<self.n_max:
            self.pontos = -1
        
    def impr_graf(self,estados,acoes,agent,rodada,n_rodadas,n_valid):
        with torch.no_grad():
            erros = deque(maxlen = n_valid)
            pontos = deque(maxlen = n_valid)
            passos = deque(maxlen = n_valid)
            for j in range(n_valid):
                saida_acao = deque()    
                saida = deque()
                pontos_sol = 0  
                erros_sol = 0
                self.inicial()
                while not self.reset:
                    acao = agent.act_target(self.entrada_agente,False)
                    self.passo(acao)
                    saida.append(self.y)
                    saida_acao.append(acao[0])
                    erros_sol += ((abs(self.y[0])+abs(self.y[2])+abs(self.y[4]))/3)
                    pontos_sol += self.pontos               
                passos.append(self.i)            
                pontos.append(pontos_sol)
                erros.append(erros_sol)
            passos_media = np.mean(passos)
            erro_media = np.mean(erros)/passos_media
            ponto_media = np.mean(pontos)
    
            
            saida = np.array(saida)                
            saida_acao = np.array(saida_acao)
            
            ax1 = plt.subplot(212)
            ax1.title.set_text('Estados')
            
            ax2 = plt.subplot(221)
            ax2.title.set_text('Erro')
            ax2.grid(True)
            ax3 = plt.subplot(222)
            ax3.title.set_text('Pontos')
            ax3.grid(True)
            box = ax1.get_position()
            if not self.arrumou:
                ax1.set_position([box.x0, box.y0, box.width * 0.90, box.height])
                self.arrumou = True
           
            ax1.cla()
            ax1.grid(True)
            for estado in estados:
                ax1.plot(range(len(saida[:,estado])),saida[:,estado],linewidth=1,label=('Estado: '+self.lista_nomes[estado]))
            for acao in acoes:
                ax1.plot(range(len(saida_acao[:,acao])),saida_acao[:,acao]/TR,linewidth=1,linestyle=':',label=('Ação: '+self.lista_acoes[acao]))
            
    
            # Put a legend to the right of the current axis
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
            
            ax2.scatter(rodada,erro_media,color="blue", marker="+", s = 5)              
            ax3.scatter(rodada,ponto_media,color = 'red', marker="+", s = 5)              
    
            
            plt.show()
            plt.pause(1)
            
            sys.stdout.write('\rProgresso: %i %% Passos %.2f Exploração: %.2f Pontos: %.2f Erro: %.2f \n' %(rodada/n_rodadas*100, passos_media, agent.sigma, ponto_media,erro_media))
