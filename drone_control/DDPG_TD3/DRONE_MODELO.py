from scipy import integrate
import numpy as np
from QUATERNION_EULER import Euler2Q, Q2Euler, dQ, Qrot
from collections import deque

## PARAMETROS DO QUADROTOR ##

# Massa e Gravidade
M, G = 1.03, 9.82
# Constantes do Motor - Empuxo e Momento
B_T = 1.435e-5
B_M = 2.4086e-7
# Momentos de Inercia nos eixos
J_XX, J_YY, J_ZZ = 16.83e-3, 16.83e-3, 28.34e-3
# Distancia do motor até o CG
D = 0.26

## INICIALIZACAO DOS PESOS ##
PESO_POSICAO = 15
PESO_VELOCIDADE = 4
PESO_ANGULO = 2
PESO_VELOCIDADE_ANG = 1
PESO_CONTROLE = 0.05
PESO_DIFF_CONTROLE = 0.5


##SHAPING##
PESO_ANG_SHAPE = PESO_ANGULO*2
PESO_VEL_SHAPE = PESO_VELOCIDADE*0.5
PESO_CONT_SHAPE = PESO_CONTROLE*0.05
P_V = 0
P_A = 5
P_F = 0
P_C = 0.3

##VALOR DE ERRO FINAL##
E_FINAL = 0.1

## BOUNDING BOXES
BB_POS = 5
BB_VEL = 10
BB_CONTROLE = 9
BB_ANG = np.pi/2


class DinamicaDrone():


    def __init__(self, passo_t, n_max, t, debug=0):
        self.t = t
        self.debug = debug
        self.cond_bb = np.array([BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_ANG, BB_ANG, 4,
                                 BB_VEL, BB_VEL, BB_VEL,
                                 n_max])
        self.entrada_hist = deque(maxlen=self.t)
        self.estados = 13
        self.y_0 = np.zeros(self.estados)
        self.tam_his = self.estados+4
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])
        self.n_max = n_max+self.t
        self.passo_t = passo_t
        self.entrada_agente = np.zeros([self.t, self.estados])

    def seed(self, seed):
        np.random.seed(seed)
    
    def FM_2_W(self,F,M):
        w = 1        
        return w
        
        
    
    def eq_drone(self, t, x, entrada):
        f_in= entrada[0]*6+M*G
        m_in = entrada[1:4]*0.8
        # w = self.FM_2_W(F,M)
        
        # f1 = B_T[0]*w[0]**2
        # f2 = B_T[1]*w[1]**2
        # f3 = B_T[2]*w[2]**2
        # f4 = B_T[3]*w[3]**2

        # m1 = B_M[0]*w[0]**2
        # m2 = B_M[1]*w[1]**2
        # m3 = B_M[2]*w[2]**2
        # m4 = B_M[3]*w[3]**2

        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]

        q = np.array([[q0, q1, q2, q3]]).T
        q = q/np.linalg.norm(q)

        F = np.array([[0, 0, f_in]]).T
        self.F_VEC = np.dot(Qrot(q), F)

        accel_w_xx = m_in[0]/(J_XX)
        accel_w_yy = m_in[1]/(J_YY)
        accel_w_zz = m_in[2]/(J_ZZ)

        W = np.array([[x[10]],
                      [x[11]],
                      [x[12]]])

        V_q = dQ(W, q)


        dq0=V_q[0]
        dq1=V_q[1]
        dq2=V_q[2]
        dq3=V_q[3]

        vel_x = x[1]
        accel_x = self.F_VEC[0]/M

        vel_y = x[3]
        accel_y = self.F_VEC[1]/M

        vel_z = x[5]
        accel_z = self.F_VEC[2]/M-G
        # Resultado da integral em ordem:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        return np.array([vel_x, accel_x,
                         vel_y, accel_y,
                         vel_z, accel_z,
                         dq0, dq1, dq2, dq3,
                         accel_w_xx, accel_w_yy, accel_w_zz])

    def inicial(self):
        self.resolvido = 0
        self.reset = False
        self.i = 0
        
        self.ang = np.random.rand(3)-0.5
        Q_in = Euler2Q(self.ang[0], self.ang[1], self.ang[2])
        self.y_0[0:6] = (np.random.rand(6)-0.5)*BB_POS
        self.y_0[6:10]=Q_in.T
        self.y_0[10:14] = (np.random.rand(3)-0.5)*1
        self.entrada = np.zeros(4) 
        self.entrada_hist.append(self.entrada)
        self.entrada_agente = np.zeros(self.t*self.tam_his)
        self.entrada_agente = np.roll(self.entrada_agente, -self.estados)
        self.entrada_agente[-self.tam_his:] = np.concatenate((self.y_0, self.entrada))
        for i in range(self.t):
            self.passo(self.entrada)
        return self.entrada_agente

    def passo(self, acao):
        if self.reset:
            print('\n\nCuidado, voce está chamando env.passo() com a flag RESET ativada. Chame env.inicial() para resetar o sistema.\n\n')
        self.i += 1
        self.entrada = acao
        self.entrada_hist.append(self.entrada)
        self.saida = integrate.solve_ivp(self.eq_drone, (0, self.passo_t), self.y_0, args=(self.entrada,))
        self.saida = self.saida.y
        self.y = np.transpose(self.saida[:, -1])
        q = np.array([self.y[6:10]]).T
        q = q/np.linalg.norm(q)
        phi, theta, psi = Q2Euler(q)
        self.ang = np.array([phi, theta, psi])
        self.entrada_agente = np.roll(self.entrada_agente, -self.estados)
        self.entrada_agente[-self.tam_his:] = np.concatenate((self.entrada, self.y))
        self.reset_fun()
        self.pontos_fun(debug=self.debug)
        self.y_0 = self.y
        return self.entrada_agente, self.pontos, self.reset

    def reset_fun(self):
        cond_x = np.concatenate((self.y[0:6], self.ang, self.y[-3:], np.array([self.i])))
        for x, c in zip(np.abs(cond_x), self.cond_bb):
            if  x >= c:
                self.reset = True



    def pontos_fun(self, debug=0):

        ## REWARD SHAPING ##

        #POSICAO
        p_p = -np.square(np.linalg.norm(self.y[0:5:2]))
        #ANGULO
        p_a = -np.square(np.sin(self.ang[2]/2)*P_A)
        
        #PENALIDADE CONTROLE ABSOLUTO
        P_CONTROLE = -np.sum(np.square(self.entrada))*P_C
        #PENALIDADE CONTROLE FORA DA MEDIA
        P_CONTROLE_D = -np.sum(np.square(self.entrada-np.mean(self.entrada_hist,0)))*P_C*5
        
        ## SOMA DOS PONTOS ##
        pontos = p_p + p_a + P_CONTROLE + P_CONTROLE_D

        #TESTE DE SOLUCAO
        est_final_parcial = 3*(E_FINAL**2)
        est_final = 12*(E_FINAL**2)
        est_teste = np.sum(np.square(np.concatenate((self.y[0:6],self.y[-3:],self.ang[0:2],[np.sin(self.ang[2]/2)]))))
        
        #ERRO DE POSICAO ABAIXO DA TOLERANCIA
        if np.sum(np.square(self.y[0:5:2])) < est_final_parcial:
            pontos += 1

            #ERRO DE VELOCIDADE ABAIXO DA TOLERANCIA
            if np.sum(np.square(self.y[1:6:2])) < est_final_parcial:
                pontos += 0.2
            
            #ERRO DE ANGULO ABAIXO DA TOLERANCIA
            if np.sum(np.square(self.ang[0:2])) < est_final_parcial:
                pontos += 0.2   
                
            #ERRO PSI ABAIXO DA TOLERANCIA
            if np.sin(self.ang[2]/2)**2 < E_FINAL**2:
                pontos += 0.2
        
        if est_teste < est_final:
            pontos += 1
            self.resolvido = 1

        if self.reset:
            self.pontos = pontos - 50
        else:
            self.pontos = pontos
        if debug:
            print('\n---Debug---')
            print('Posicao Atual: ' +str(self.y[0:5:2]))
            print('Psi Atual: %.2f Ponto: %.2f' %(self.ang[2],p_a))
            print('Velocidade: '+ str(self.y[1:6:2]))
            print('Força: '+str(self.F_VEC))
            print('Entrada: '+str(self.entrada))
            print('TOTAL: %.2f Direçao: %.2f%% Psi: %.2f%% Cont: %.2f%% D Cont: %.2f%%' %(pontos, p_p/pontos*100, p_a/pontos*100, P_CONTROLE/pontos*100, P_CONTROLE_D/pontos*100))
            print('---Fim Debug---')