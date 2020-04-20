from scipy import integrate
import numpy as np
from QUATERNION_EULER import Euler2Q, Q2Euler, dQ, Qrot

## PARAMETROS DO QUADROTOR ##

# Massa e Gravidade
M, G = 1.2, 10
# Constantes do Motor - Empuxo e Momento
B_T = 1.875e-7
B_M = 1.875e-8
# Momentos de Inercia nos eixos
J_XX, J_YY, J_ZZ = 8.3e-3, 8.3e-3, 15.5e-3
# Distancia do motor até o CG
D = 0.25

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
P_P = 0.3
P_V = 1
P_A = 0.1
P_F = 0.1
P_C = 0.0001

##VALOR DE ERRO FINAL##
E_FINAL = 0.1

## BOUNDING BOXES
BB_POS = 5
BB_VEL = 10
BB_CONTROLE = 9
BB_ANG = np.pi/2

## VALORES ANTERIORES
T = 2 #amostras anteriores de entrada na rede
class DinamicaDrone():


    def __init__(self, passo_t, n_max, debug=0):
        self.debug = debug
        self.cond_bb = np.array([BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_ANG, BB_ANG, 4,
                                 BB_VEL, BB_VEL, BB_VEL,
                                 n_max])
        self.estados = 13
        self.y_0 = np.zeros(self.estados)
        self.tam_his = self.estados+4
        self.reset = True
        self.i = 0
        self.y = np.zeros([self.estados*2])
        self.n_max = n_max+T
        self.passo_t = passo_t
        self.entrada_agente = np.zeros([T, self.estados])

    def seed(self, seed):
        np.random.seed(seed)
    
    def FM_2_W(self,F,M):
        w = 1        
        return w
        
        
    
    def eq_drone(self, t, x, entrada):
        f_in= entrada[0]+12
        m_in = entrada[1:4]
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

        accel_w_xx = m_in[0]/J_XX
        accel_w_yy = m_in[1]/J_YY
        accel_w_zz = m_in[2]/J_ZZ

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
        self.entrada[0] = 12        
        self.entrada_agente = np.zeros(T*self.tam_his)
        self.entrada_agente = np.roll(self.entrada_agente, -self.estados)
        self.entrada_agente[-self.tam_his:] = np.concatenate((self.y_0, self.entrada))
        for i in range(T):
            self.passo(self.entrada)
        return self.entrada_agente

    def passo(self, acao):
        if self.reset:
            print('\n\nCuidado, voce está chamando env.passo() com a flag RESET ativada. Chame env.inicial() para resetar o sistema.\n\n')
        self.i += 1
        self.entrada_anterior = self.entrada
        self.entrada = acao
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
        p_p = np.exp(-(np.linalg.norm(self.y[0:5:2]))*P_P)*3

        target_v_x = np.clip(-self.y[0]*P_V, -4, 4)
        target_v_y = np.clip(-self.y[2]*P_V, -4, 4)
        target_v_z = np.clip(-self.y[4]*P_V, -4, 4)

        target_FX = np.clip((target_v_x-self.y[1])*P_A, -6, 6)
        target_FY = np.clip((target_v_y-self.y[3])*P_A, -6, 6)
        target_FZ = 12+np.clip((target_v_z-self.y[5])*P_A, -6, 6)



        P_T_X = -((target_FX-self.F_VEC[0])/6)**2*P_F
        P_T_Y = -((target_FY-self.F_VEC[1])/6)**2*P_F
        P_T_Z = -((target_FZ-self.F_VEC[2])/6)**2*P_F

        P_CONTROLE = -np.sum(np.square(self.entrada))*P_C
        
        p_target = P_T_X+P_T_Y+P_T_Z

        ## SOMA DOS PONTOS ##
        pontos = p_p + p_target + P_CONTROLE

        #TESTE DE SOLUCAO
        for x in self.y[0:5:2]:
            if abs(x) < abs(E_FINAL):
                pontos += 1

        if np.sum(np.square(self.y[0:6])) < np.sum(np.square(np.ones(6)*E_FINAL)):
            pontos += 6
            self.resolvido = 1

        if self.reset:
            self.pontos = pontos - 100
        else:
            self.pontos = pontos
        if debug:
            print('\n---Debug---')
            print('Posicao Atual: ' +str(self.y[0:5:2]))
            print('Alvo de Velocidade: ' + str([target_v_x, target_v_y, target_v_z]))
            print('Velocidade: '+ str(self.y[1:6:2]))
            print('Alvo de Força: '+str([target_FX, target_FY, target_FZ]))
            print('Força: '+str(self.F_VEC))
            print('Entrada: '+str(self.entrada))
            print('TOTAL: %.2f P Direçao: %.2f%% P FX: %.2f%% P FY: %.2f%% P FZ: %.2f%% Cont: %.2f%%' %(pontos, p_p/pontos*100, P_T_X/pontos*100, P_T_Y/pontos*100, P_T_Z/pontos*100, P_CONTROLE/pontos*100))
            print('---Fim Debug---')