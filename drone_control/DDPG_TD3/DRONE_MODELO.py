from scipy import integrate
import numpy as np
from QUATERNION_EULER import Euler2Q, Q2Euler, dQ, Qrot

## PARAMETROS DO QUADROTOR ##

# Massa e Gravidade
m, g = 1.2, 10
# Constantes do Motor - Empuxo e Momento
b = 1.875e-7
B = 1.875e-8
# Momentos de Inercia nos eixos
J_xx,J_yy,J_zz = 8.3e-3,8.3e-3,15.5e-3
# Distancia do motor até o CG
d = 0.25

## INICIALIZACAO DOS PESOS ##
PESO_POSICAO = 15
PESO_VELOCIDADE = 4
PESO_ANGULO = 2
PESO_VELOCIDADE_ANG = 1
PESO_CONTROLE = 0.5
PESO_DIFF_CONTROLE = 0.5
 

##SHAPING##
PESO_ANG_SHAPE = PESO_ANGULO*2
PESO_VEL_SHAPE = PESO_VELOCIDADE*0.5
PESO_CONT_SHAPE = PESO_CONTROLE*0.05
P_V = 0.1
P_A = 0.1



## BOUNDING BOXES
BB_POS = 300
BB_VEL = 30
BB_CONTROLE = 9
BB_ANG = np.pi/3

## VALORES ANTERIORES
T = 2 #amostras anteriores de entrada na rede
class DinamicaDrone():


    def __init__(self,passo_t,n_max):
        self.resolvido = 0
        self.cond_bb = np.array([BB_POS,BB_VEL,BB_POS,BB_VEL,BB_POS,BB_VEL,BB_ANG,BB_ANG,4,BB_VEL,BB_VEL,BB_VEL,n_max])        
        self.estados = 13
        self.tam_his = self.estados+4
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])
        self.n_max = n_max+T
        self.passo_t = passo_t
        self.entrada_agente = np.zeros([T,self.estados])

    def seed(self,seed):
        np.random.seed(seed)

    def eq_drone(self,t,x,entrada):
        f1 = entrada[0]
        f2 = entrada[1]
        f3 = entrada[2]
        f4 = entrada[3]

        m1 = B/b*f1
        m2 = B/b*f2
        m3 = B/b*f3
        m4 = B/b*f4

        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]

        q = np.array([[q0,q1,q2,q3]]).T
        q = q/np.linalg.norm(q)

        F = np.array([[0,0,(f1+f2+f3+f4)]]).T
        self.F_VEC = np.dot(Qrot(q),F)

        accel_w_xx = (f1-f3)*d/J_xx
        accel_w_yy = (f2-f4)*d/J_yy
        accel_w_zz = (m1+m3-m2-m4)/J_zz

        W = np.array([[x[10]],
                      [x[11]],
                      [x[12]]])

        V_q = dQ(W,q)


        dq0=V_q[0]
        dq1=V_q[1]
        dq2=V_q[2]
        dq3=V_q[3]
        
        self.F_VEC = self.F_VEC - np.array([[0,0,g*m]]).T

        vel_x = x[1]
        accel_x = self.F_VEC[0]/m

        vel_y = x[3]
        accel_y = self.F_VEC[1]/m

        vel_z = x[5]
        accel_z = self.F_VEC[2]/m
        # Resultado da integral em ordem:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        return np.array([vel_x,accel_x,vel_y,accel_y,vel_z,accel_z,dq0,dq1,dq2,dq3,accel_w_xx,accel_w_yy,accel_w_zz])

    def inicial(self):
        ang_in = np.random.rand(3)-0.5
        Q_in = Euler2Q(ang_in[0],ang_in[1],ang_in[2])
        self.y_0 = (np.random.rand(self.estados)-0.5)*10
        self.y_0[10:14]=self.y_0[10:14]/5
        self.y_0[6:10]=Q_in.T
        self.reset = False
        self.i = 0
        self.entrada = np.ones(4)*3
        self.N_RESOLVIDO = 0
        sin_cos_entrada = np.array([np.sin(ang_in[0]),np.cos(ang_in[0]),np.sin(ang_in[1]),np.cos(ang_in[1]),np.sin(ang_in[2]),np.cos(ang_in[2])])
        self.entrada_agente = np.zeros(T*self.tam_his)
        self.entrada_agente = np.roll(self.entrada_agente,-self.estados)
        self.entrada_agente[-self.tam_his:] = np.concatenate((self.y_0,self.entrada))
        for i in range(T):
            self.passo(self.entrada)
        return self.entrada_agente

    def passo(self,acao):
        if self.reset:
            print('\n\nCuidado, voce está chamando env.passo() com a flag RESET ativada. Chame env.inicial() para resetar o sistema.\n\n')
        self.i += 1
        self.entrada_anterior = self.entrada
        self.entrada = acao
        self.saida = (integrate.solve_ivp(self.eq_drone,(0,self.passo_t),self.y_0, args=[self.entrada]))
        self.saida = self.saida.y
        self.y = np.transpose(self.saida[:,-1])
        q = np.array([self.y[6:10]]).T
        q = q/np.linalg.norm(q)
        self.phi,self.theta,self.psi = Q2Euler(q)
        self.ang = np.array([self.phi,self.theta,self.psi])
        sin_cos_entrada = np.array([np.sin(self.phi),np.cos(self.phi),np.sin(self.theta),np.cos(self.theta),np.sin(self.psi),np.cos(self.psi)])
        self.entrada_agente = np.roll(self.entrada_agente,-self.estados)
        self.entrada_agente[-self.tam_his:] = np.concatenate((self.entrada,self.y))
        self.reset_fun()
        self.pontos_fun()
        self.y_0 = self.y
        return self.entrada_agente, self.pontos, self.reset

    def reset_fun(self):
        cond_x = np.concatenate((self.y[0:6],self.ang,self.y[-3:],np.array([self.i])))
        for x,c in zip(np.abs(cond_x),self.cond_bb):
            if  x >= c:
                self.reset = True

        

    def pontos_fun(self):
      
        
        ## REWARD SHAPING ##
        
        #POSICAO
        p_p = 0
        p_p = np.linalg.norm(-self.y_0[0:5:2])-np.linalg.norm(-self.y[0:5:2])
        
        
        target_v_x = -abs(-self.y[0]*P_V-self.y[1])
        target_v_y = -abs(-self.y[2]*P_V-self.y[3])
        target_v_z = -abs(-self.y[4]*P_V-self.y[5])
        
        target_phi = -abs((-self.y[2]*P_V-self.y[4])*P_A+self.ang[0])
        target_theta = -abs((-self.y[0]*P_V-self.y[3])*P_A+self.ang[1])
        target_psi = -abs(self.ang[2])
        
        p_target = target_v_x+target_v_y+target_v_z+target_phi+target_theta+target_psi
        
        ## SOMA DOS PONTOS ##   
        pontos = p_p + p_target#+ p_v + p_ang + p_v_ang + p_cont 
        
        #TESTE DE SOLUCAO
        for x in self.y[0:5:2]:
            if abs(x) < abs(0.1):
                pontos += 3
        
        if np.sum(np.square(self.y[0:6])) < np.sum(np.square(np.ones(6)*0.1)):
            pontos += 25
            print('\n### Resolvido! ### \n')
            self.resolvido += 1
            if self.resolvido > 30:
                pontos += 500
                self.reset = True
            
        if self.reset:
            self.pontos = pontos - 5
        else:
            self.pontos = pontos