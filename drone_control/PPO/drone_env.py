from scipy import integrate
import numpy as np
from QUATERNION_EULER import Euler2Q, Q2Euler, dQ, Qrot
from collections import deque
from numpy.linalg import norm
## QUADROTOR PARAMETERS ##

# QUADROTOR MASS AND GRAVITY VALUE
M, G = 1.03, 9.82

# ELETRIC MOTOR THRUST AND MOMENT
B_T = 1.435e-5
B_M = 2.4086e-7

# INERTIA MATRIX
J = np.array([[16.83e-3, 0, 0],
              [0, 16.83e-3, 0],
              [0, 0, 28.34e-3]])

# ELETRIC MOTOR DISTANCE TO CG
D = 0.26


## CONTROL REWARD PENALITIES ##
P_C = 0.1
P_C_D = 0.5

## TARGET STEADY STATE ERROR ##
TR = [0.01,0.05,0.1]
TR_P = [40,20,10]

## BOUNDING BOXES ##
BB_POS = 5 
BB_VEL = 10
BB_CONTROL = 9
BB_ANG = np.pi/2


class drone():

    def __init__(self, t_step, n, euler=0, deep_learning_flag=0, T=1, debug=0):
        
        """"
        inputs:
            t_step: integration time step 
            n: max timesteps
            euler: flag to set the states return in euler angles, if off returns quaternions
            deep learning:
                deep learning flag: If on, changes the way the env. outputs data, optimizing it to deep learning use.
                T: Number of past history of states/actions used as inputs in the neural network
                debug: If on, prints a readable reward funcion, step by step.
        
        """
        
        
        self.T = T
        self.debug = debug
        self.bb_cond = np.array([BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_ANG, BB_ANG, 4,
                                 BB_VEL, BB_VEL, BB_VEL])
        self.action_hist = deque(maxlen=self.T)
        self.state_size = 13
        self.action_size = 4
        self.y_0 = np.zeros(self.state_size)
        self.hist_size = self.state_size+self.action_size
        self.done = True
    
        self.n = n+self.T
        self.t_step = t_step
        self.deep_learning_flag = deep_learning_flag
        self.euler_flag = euler
        
    def seed(self, seed):
        
        """"
        Set random seeds for reproducibility
        """
        
        np.random.seed(seed)
    
    ## FUNCAO AINDA A SER FEITA - TRANSFORMA AS FORCAS EM VELOCIDADES ANGULARES ##
    def FM_2_W(self,F,M):
        w = 1        
        return w
        
        
    
    def drone_eq(self, t, x, action):
        
        """"
        Main differential equation, not used directly by the user, rather used in the step function.
        """
        
        f_in= action[0]*6+M*G
        m_in = action[1:4]*0.8
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

        f_body = np.array([[0, 0, f_in]]).T
        self.f_inertial = np.dot(Qrot(q), f_body)

        accel_w_xx = m_in[0]/(J[0,0])
        accel_w_yy = m_in[1]/(J[1,1])
        accel_w_zz = m_in[2]/(J[2,2])

        W = np.array([[x[10]],
                      [x[11]],
                      [x[12]]])

        V_q = dQ(W, q)


        dq0=V_q[0]
        dq1=V_q[1]
        dq2=V_q[2]
        dq3=V_q[3]

        vel_x = x[1]
        accel_x = self.f_inertial[0]/M

        vel_y = x[3]
        accel_y = self.f_inertial[1]/M

        vel_z = x[5]
        accel_z = self.f_inertial[2]/M-G
        # Resultado da integral em ordem:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        return np.array([vel_x, accel_x,
                         vel_y, accel_y,
                         vel_z, accel_z,
                         dq0, dq1, dq2, dq3,
                         accel_w_xx, accel_w_yy, accel_w_zz])

    def reset(self,det_state=0):
        
        """""
        inputs:
            det_state: 
                if == 0 randomized initial state
                else det_state is the actual initial state, depending on the euler flag
        outputs:
            previous_state: system's initial state
        """""
        
        self.solved = 0
        self.done = False
        self.i = 0   
        self.prev_shaping = None
        self.previous_state = np.zeros(self.state_size)
        
        if det_state:
            if self.euler_flag:
                phi, theta, psi = det_state[6],det_state[7],det_state[8]    
                quaternions = Euler2Q(phi,theta,psi).flatten()
                self.previous_state = np.concatenate((det_state[0:6],quaternions,det_state[-3:]))           
            else:
                self.previous_state = det_state
        else:
            self.ang = np.random.rand(3)-0.5
            Q_in = Euler2Q(self.ang[0], self.ang[1], self.ang[2])
            self.previous_state[0:6] = (np.random.rand(6)-0.5)*BB_POS
            self.previous_state[6:10] = Q_in.T
            self.previous_state[10:14] = (np.random.rand(3)-0.5)*1


        self.deep_learning_input = np.zeros(self.T*self.hist_size)
        self.deep_learning_input[-self.state_size:] = self.previous_state
        
        for i in range(self.T-1):
            self.action = np.zeros(4) 
            self.action_hist.append(self.action)
            self.step(self.action)
        
        if self.deep_learning_flag:
            return self.deep_learning_input
        else:
            if self.euler_flag:
                phi, theta, psi = Q2Euler(np.array([self.previous_state[6:10]]).T)
                ang = np.array([phi,theta,psi])
                return np.concatenate((self.previous_state[0:6],ang,self.previous_state[-3:]))
            else:
                return self.previous_state




    def step(self, action):
        
        """""
        inputs:
            action: action to be applied on the system
        outputs:
            state: system's state in t+t_step actuated by the action
            done: False, else the system has breached any bounding box, exceeded maximum timesteps, or reached goal.
        """""
        
        if self.done:
            print('\n----WARNING----\n done flag is TRUE, reset the environment with environment.reset() before using environment.step()\n')
        self.i += 1
        self.action = np.clip(action,-1,1)
        self.action_hist.append(self.action)
        
        
        
        self.y = (integrate.solve_ivp(self.drone_eq, (0, self.t_step), self.previous_state, args=(self.action, ))).y
        self.state = np.transpose(self.y[:, -1])
        
        
        q = np.array([self.state[6:10]]).T
        q = q/np.linalg.norm(q)
        phi, theta, psi = Q2Euler(q)
        self.ang = np.array([phi, theta, psi])
        
        
        self.deep_learning_input = np.roll(self.deep_learning_input, -self.hist_size)
        self.deep_learning_input[-self.hist_size:] = np.concatenate((self.action, self.state))
        self.done_condition()
        self.previous_state = self.state
        
        if self.deep_learning_flag:
            self.reward_function(debug=self.debug)    
            return self.deep_learning_input, self.reward, self.reset
        else:
            if self.euler_flag:
                return np.concatenate((self.state[0:6],self.ang,self.state[-3:])), self.done
            else:
                return self.state, self.done

    def done_condition(self):
        
        """""
        Checks if bounding boxes done condition has been met
        """""
        
        cond_x = np.concatenate((self.state[0:6], self.ang, self.state[-3:]))
        for x, c in zip(np.abs(cond_x), self.bb_cond):
            if  x >= c:
                self.done = True


    def reward_function(self, debug=0):
        self.reward = 0
        
        position = self.state[0:5:2]
        velocity = self.state[1:6:2]
        euler_angles = self.ang
        psi = self.ang[2]
        body_ang_vel = self.state[-3:]
        action = self.action
        action_hist = self.action_hist
        
        shaping = 100*(-norm(position/BB_POS) - norm(velocity/BB_VEL) - abs(psi/4))
        
        #CASCADING REWARDS
        r_state = np.concatenate((position,[psi]))
        
        for TR_i,TR_Pi in zip(TR,TR_P): 
            if norm(r_state) < norm(np.ones(len(r_state))*TR_i):
                shaping += TR_Pi
                if norm(euler_angles) < norm(np.ones(3)*TR_i*2):
                    shaping += TR_Pi
                if norm(velocity) < norm(np.ones(3)*TR_i):
                    shaping += TR_Pi
                break
        
        if self.prev_shaping is not None:
            self.reward = shaping - self.prev_shaping
       
        #ABSOLUTE CONTROL PENALITY
        abs_control = -np.sum(np.square(action)) * P_C
        #AVERAGE CONTROL PENALITY        
        avg_control = -np.sum(np.square(action - np.mean(action_hist,0))) * P_C_D
        
        ## TOTAL REWARD SHAPING ##
        self.reward += + abs_control + avg_control
        
        #SOLUTION ACHIEVED?
        target_state = 12*(TR[0]**2)
        current_state = np.sum(np.square(np.concatenate((position, velocity, euler_angles, body_ang_vel))))      
        
        if current_state < target_state:
            self.reward = +500
            self.rsolved = 1
            self.done = True 
            
        elif self.i >= self.n and not self.done:
            self.reward = self.reward
            self.reset = True
            self.resolvido = 0
            
        elif self.done:
            self.reward = -200
            
        if debug and self.i%debug==0 and self.prev_shaping is not None:
            print('\n---Starting Debug---')
            print('Pos: ' + str(position) + '\t Velocity: '+ str(velocity))
            print('Euler: ' +str(euler_angles) + '\t Body Ang Velocity: ' + str(body_ang_vel))
            print('Action: '+str(self.input))
            print('Timestep: ' + str(self.i))
            print('Reward: %.2f \t Prev Shaping: %.2f \t Shaping: %.2f \n ABS Cont: %.2f \t AVG Cont: %.2f' %(self.reward, self.prev_shaping, shaping, abs_control, avg_control))
            print('---Debug End---')
        self.prev_shaping = shaping