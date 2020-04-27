from scipy import integrate
import numpy as np
from quaternion_euler_utility import euler_quat, quat_euler, deriv_quat, quat_rot_mat
from collections import deque
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import vpython as visual
import PIL.ImageGrab
# from vpython.no_notebook import stop_server

"""""
QUADROTOR ENVIRONMENT
DEVELOPED BY: 
    RAFAEL COSTA FERNANDES
    PROGRAMA DE PÓS GRADUAÇÃO EM ENGENHARIA MECÂNICA, UNIVERSIDADE FEDERAL DO ABC
    SP - SANTO ANDRÉ - BRASIL

FURTHER DOCUMENTATION ON README.MD
"""""

## QUADROTOR PARAMETERS ##

## SIMULATION BOUNDING BOXES ##
BB_POS = 5 
BB_VEL = 10
BB_CONTROL = 9
BB_ANG = np.pi/2

# QUADROTOR MASS AND GRAVITY VALUE
M, G = 1.03, 9.82

# AIR DENSITY
RHO = 1.2041

#DRAG COEFFICIENT
C_D = 1.1

# ELETRIC MOTOR THRUST AND MOMENT
K_F = 1.435e-5
K_M = 2.4086e-7
I_R = 5e-5
T2WR = 2

## INDIRECT CONTROL CONSTANTS ##
IC_THRUST = 6
IC_MOMENTUM = 0.8


# INERTIA MATRIX
J = np.array([[16.83e-3, 0, 0],
              [0, 16.83e-3, 0],
              [0, 0, 28.34e-3]])

# ELETRIC MOTOR DISTANCE TO CG
D = 0.26

#PROJECTED AREA IN X_b, Y_b, Z_b
BEAM_THICKNESS = 0.05
A_X = BEAM_THICKNESS*2*D
A_Y = BEAM_THICKNESS*2*D
A_Z = BEAM_THICKNESS*2*D*2
A = np.array([[A_X,A_Y,A_Z]]).T


## REWARD PARAMETERS ##

# CONTROL REWARD PENALITIES #
P_C = 0.2
P_C_D = 0.3

## TARGET STEADY STATE ERROR ##
TR = [0.1]
TR_P = [10]


class quad():

    def __init__(self, t_step, n, euler=0, direct_control=0, deep_learning=0, T=1, debug=0):
        current = visual.canvas.get_selected()
        current.delete()
        
        """"
        inputs:
            t_step: integration time step 
            n: max timesteps
            euler: flag to set the states return in euler angles, if off returns quaternions
            deep learning:
                deep learning flag: If on, changes the way the env. outputs data, optimizing it to deep learning use.
                T: Number of past history of states/actions used as inputs in the neural network
                debug: If on, prints a readable reward funcion, step by step, for a simple reward weight debugging.
        
        """
        
        
        self.T = T                                              #Deep Learning Time History
        self.debug = debug                                      #Reward Function debug flag
        self.bb_cond = np.array([BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_POS, BB_VEL,
                                 BB_ANG, BB_ANG, 4,
                                 BB_VEL, BB_VEL, BB_VEL])       #Bounding Box Conditions Array
        
        
        self.state_size = 13                                    #Quadrotor states dimension
        self.action_size = 4                                    #Quadrotor action dimension
        
        
        self.hist_size = self.state_size+1+self.action_size     #Deep Learning step dimension
        self.deep_learning_in_size = self.hist_size*self.T      #Deep Learning whole dimension (T steps)
        
        self.action_hist = deque(maxlen=self.T)                 #Action history initialization, to use with average action penality         

        self.done = True                                        #Env done Flag
    
        self.n = n+self.T                                       #Env Maximum Steps
        self.t_step = t_step
        self.deep_learning_flag = deep_learning
        self.euler_flag = euler
        self.direct_control_flag = direct_control
        
        self.inv_j = np.linalg.inv(J)
        self.zero_control = np.ones(4)*(2/T2WR - 1)             #Neutral Action (used in reset and absolute action penality) 
        
    def seed(self, seed):
        
        """"
        Set random seeds for reproducibility
        """
        
        np.random.seed(seed)       
    
    
    
    def f2w(self,f,m):
        """""
        Translates F (Thrust) and M (Body x, y and z moments) into eletric motor angular velocity (rad/s)
        """""
        x = np.array([[1, 1, 1, 1],
                      [-1/D, 0, 1/D, 0],
                      [0, 1/D, 0, -1/D],
                      [-K_F/K_M, +K_F/K_M, -K_F/K_M, +K_F/K_M]])      
        
        y = np.array([f, m[0,0], m[1,0], m[2,0]])
        
        u = np.linalg.solve(x,y)
        u = np.clip(u,0,T2WR*M*G/4)
        
        w_1 = np.sqrt(u[0]/K_F)
        w_2 = np.sqrt(u[1]/K_F)
        w_3 = np.sqrt(u[2]/K_F)
        w_4 = np.sqrt(u[3]/K_F)        
        w = np.array([w_1,w_2,w_3,w_4])

        FM_new = np.dot(x,u)
        
        F_new = FM_new[0]
        M_new = FM_new[1:4]
        
        return u, w, F_new, M_new
        
    def f2F(self,f_action):
        f = (f_action+1)*T2WR*M*G/8

        
        w = np.array([[np.sqrt(f[0]/K_F)],
                      [np.sqrt(f[1]/K_F)],
                      [np.sqrt(f[2]/K_F)],
                      [np.sqrt(f[3]/K_F)]])
        
        F_new = np.sum(f)
        M_new = np.array([[(f[2]-f[0])*D],
                          [(f[1]-f[3])*D],
                          [(-f[0]+f[1]-f[2]+f[3])*K_M/K_F]])
        return w, F_new, M_new
    
    def drone_eq(self, t, x, action):
        
        """"
        Main differential equation, not used directly by the user, rather used in the step function integrator.
        Dynamics based in: 
            MODELAGEM DINÂMICA E CONTROLE DE UM VEÍCULO AÉREO NÃO TRIPULADO DO TIPO QUADRIRROTOR 
            by ALLAN CARLOS FERREIRA DE OLIVEIRA
            BRASIL, SP-SANTO ANDRÉ, UFABC - 2019
        Incorporates:
            Drag Forces, Gyroscopic Forces
            In indirect mode: Force clipping (preventing motor shutoff and saturates over Thrust to Weight Ratio)
            In direct mode: maps [-1,1] to forces [0,T2WR*G*M/4]
        """
        w, f_in, m_action = self.f2F(action)
        
        
        #BODY INERTIAL VELOCITY                
        vel_x = x[1]
        vel_y = x[3]
        vel_z = x[5]
        
        #QUATERNIONS
        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]
        
        #BODY ANGULAR VELOCITY
        w_xx = x[10]
        w_yy = x[11]
        w_zz = x[12]      

        #QUATERNION NORMALIZATION (JUST IN CASE)
        q = np.array([[q0, q1, q2, q3]]).T
        q = q/np.linalg.norm(q)
        
        # DRAG FORCES ESTIMATION (BASED ON BODY VELOCITIES)
        v_inertial = np.array([[vel_x, vel_y, vel_z]]).T
        v_body = np.dot(quat_rot_mat(q).T, v_inertial)
        f_drag = -0.5*RHO*C_D*np.multiply(A,np.multiply(abs(v_body),v_body))
        
        # DRAG MOMENTS ESTIMATION (BASED ON BODY ANGULAR VELOCITIES)
        
        #Discretization over 10 steps (linear velocity varies over the body)
        d_xx = np.linspace(0,D,10)
        d_yy = np.linspace(0,D,10)
        d_zz = np.linspace(0,D,10)
        m_x = 0
        m_y = 0
        m_z = 0
        for xx,yy,zz in zip(d_xx,d_yy,d_zz):
            m_x += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(xx*w_xx)*(xx*w_xx))
            m_y += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(yy*w_yy)*(yy*w_yy))
            m_z += -2*RHO*C_D*BEAM_THICKNESS*D/10*(abs(zz*w_zz)*(zz*w_zz))
        
        m_drag = np.array([[m_x],
                           [m_y],
                           [m_z]])
        
        #GYROSCOPIC EFFECT ESTIMATION (BASED ON ELETRIC MOTOR ANGULAR VELOCITY)        
        
        omega_r = (-w[0]+w[1]-w[2]+w[3])[0]
        
        m_gyro = np.array([[-w_xx*I_R*omega_r],
                           [+w_yy*I_R*omega_r],
                           [0]])

        #BODY FORCES
        f_body = np.array([[0, 0, f_in]]).T+f_drag
        
        #BODY FORCES ROTATION TO INERTIAL
        self.f_inertial = np.dot(quat_rot_mat(q), f_body)
        
        #INERTIAL ACCELERATIONS        
        accel_x = self.f_inertial[0]/M        
        accel_y = self.f_inertial[1]/M        
        accel_z = self.f_inertial[2]/M-G
        
        #BODY MOMENTUM
        m_in = m_action + m_gyro + m_drag

        #INERTIAL ANGULAR ACCELERATION        
        accel_ang = np.dot(self.inv_j,m_in).flatten()
        accel_w_xx = accel_ang[0]
        accel_w_yy = accel_ang[1]
        accel_w_zz = accel_ang[2]
        
        #QUATERNION ANGULAR VELOCITY (INERTIAL)
        W = np.array([[w_xx],
                      [w_yy],
                      [w_zz]])
        self.V_q = deriv_quat(W, q).flatten()
        dq0=self.V_q[0]
        dq1=self.V_q[1]
        dq2=self.V_q[2]
        dq3=self.V_q[3]
        
        # print('Debug')
        # print(v_inertial,v_body)
        # print(f_drag,f_body,self.f_inertial)
        # print([w_xx,w_yy,w_zz])
        # print(m_drag,m_in)
        # print('Debug')
        
        # RESULTS ORDER:
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
                if euler flag is on:
                    [x, dx, y, dy, z, dz, phi, theta, psi, w_xx, w_yy, w_zz]
                if euler flag is off:
                    [x, dx, y, dy, z, dz, q_0, q_1, q_2, q_3, w_xx, w_yy, w_zz]
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
                ang = det_state[6:9]
                quaternions = Euler2Q(ang).flatten()
                self.previous_state = np.concatenate((det_state[0:6],quaternions,det_state[-3:]))           
            else:
                self.previous_state = det_state
        else:
            self.ang = np.random.rand(3)-0.5
            Q_in = euler_quat(self.ang)
            self.previous_state[0:6] = (np.random.rand(6)-0.5)*BB_POS
            self.previous_state[6:10] = Q_in.T
            self.previous_state[10:13] = (np.random.rand(3)-0.5)*1


        self.deep_learning_input = np.zeros(self.deep_learning_in_size)
        
        for i in range(self.T):
            self.action = self.zero_control
            self.action_hist.append(self.action)
            self.step(self.action)
        
        if self.deep_learning_flag:
            return self.deep_learning_input
        else:
            if self.euler_flag:
                ang = Q2Euler(np.array([self.previous_state[6:10]]).T)
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
        
        
        if self.direct_control_flag:
            u = self.action
            self.clipped_action = self.action
        else:
            f_in= action[0]*IC_THRUST+M*G
            m_action = (np.array([self.action[1:4]]).T)*IC_MOMENTUM
            u, _, f_new, m_new = self.f2w(f_in,m_action)
            #CLIPPED ACTION FOR LOGGING 
            self.clipped_action = np.array([(f_new-M*G)/IC_THRUST,
                                            m_new[0]/IC_MOMENTUM,
                                            m_new[1]/IC_MOMENTUM,
                                            m_new[2]/IC_MOMENTUM])
        
        
        self.y = (integrate.solve_ivp(self.drone_eq, (0, self.t_step), self.previous_state, args=(u, ))).y
        self.state = np.transpose(self.y[:, -1])
        
        
        q = np.array([self.state[6:10]]).T
        q = q/np.linalg.norm(q)
        self.ang = quat_euler(q)
        
        
        self.deep_learning_input = np.roll(self.deep_learning_input, -self.hist_size)
        self.deep_learning_input[-self.hist_size:] = np.concatenate((self.action, self.state[0:10], self.V_q))
        
        self.previous_state = self.state
        self.done_condition()
        
        
        if self.deep_learning_flag:
            self.reward_function(debug=self.debug)    
            return self.deep_learning_input, self.reward, self.done
        else:
            if self.euler_flag:
                return np.concatenate((self.state[0:6],self.ang,self.state[-3:])), self.done
            else:
                return self.state, self.done

    def done_condition(self):
        
        """""
        Checks if bounding boxes done condition have been met
        """""
        
        cond_x = np.concatenate((self.state[0:6], self.ang, self.state[-3:]))
        for x, c in zip(np.abs(cond_x), self.bb_cond):
            if  x >= c:
                self.done = True
        if not self.deep_learning_flag and self.i >= self.n:
            self.done = True

    def reward_function(self, debug=0):
        
        """""
        Reward Function: Working with PPO great results.
        Shaping with some ideas based on Continuous Lunar Lander v.2 gym environment:
            https://gym.openai.com/envs/LunarLanderContinuous-v2/
        
        """""
        
        
        self.reward = 0
        
        position = self.state[0:5:2]
        velocity = self.state[1:6:2]
        euler_angles = self.ang
        psi = self.ang[2]
        body_ang_vel = self.state[-3:]
        action = self.action
        action_hist = self.action_hist
        
        shaping = 100*(-norm(position/BB_POS)-norm(velocity/BB_VEL)-norm(psi/4)-0.3*norm(euler_angles[0:2]/BB_ANG))
        
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
        self.prev_shaping = shaping
        
        #ABSOLUTE CONTROL PENALITY
                   
        abs_control = -np.sum(np.square(action - self.zero_control)) * P_C
        #AVERAGE CONTROL PENALITY        
        avg_control = -np.sum(np.square(action - np.mean(action_hist,0))) * P_C_D
        
        ## TOTAL REWARD SHAPING ##
        self.reward += + abs_control + avg_control
        
        #SOLUTION ACHIEVED?
        target_state = 12*(TR[0]**2)
        current_state = np.sum(np.square(np.concatenate((position, velocity, euler_angles, body_ang_vel))))      

        if current_state < target_state:
            self.reward = +500
            self.solved = 1
            self.done = True 
            
        elif self.i >= self.n and not self.done:
            self.reward = self.reward
            self.done = True
            self.solved = 0
            
        elif self.done:
            self.reward = -200
            self.solved = 0
            
        if debug and self.i%debug==0 and self.prev_shaping is not None:
            print('\n---Starting Debug---')
            print('Pos: ' + str(position) + '\t Velocity: '+ str(velocity))
            print('Euler: ' +str(euler_angles) + '\t Body Ang Velocity: ' + str(body_ang_vel))
            print('Action: '+str(self.input))
            print('Timestep: ' + str(self.i))
            print('Reward: %.2f \t Prev Shaping: %.2f \t Shaping: %.2f \n ABS Cont: %.2f \t AVG Cont: %.2f' %(self.reward, self.prev_shaping, shaping, abs_control, avg_control))
            print('---Debug End---')
        
            
class render(): 
        
    """""
    Render Class: Saves state and time until plot function is called.
                    Optionally: Plots a 3D graph of the position, with optional target position.
    
    init input:
        print_list: 
            array of desired states to plot should look like this:
                print_list = [0, 2, 4, 5, 6]
            this states you wnat to print states 0, 2, 4 etc..
        plot_labels:
            array of names of each desired plot state, should look like this:
                plot_labels = ['x', 'y', ...]
        line_styles:
            array of line styles of each desired plot state, should look like this:
                plot_labels = [':', '--', ...]
        depth_plot_list:
            list of states and target states (optional)
            
            
    add: saves a state and a time

    clear: clear memory

    plot: plot saved states as choosen in init file  

    depth_plot: 3d plots position and target position (if required)
                target position must be in states        
    """""   
    
    def __init__(self, print_list, plot_labels, line_styles, depth_plot_list=0, animate=0):        
        plt.close('all')
        fig2d = plt.figure('States')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        self.states = []
        self.times = []
        self.print_list = print_list
        self.plot_labels = plot_labels
        self.line_styles = line_styles
        self.depth_plot_list = depth_plot_list
        
        if self.depth_plot_list:            
            fig3d = plt.figure('3D map')
            self.ax = Axes3D(fig3d)
            self.ax.set_xlabel('x (m)')
            self.ax.set_ylabel('y (m)')
            self.ax.set_zlabel('z (m)')
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        
            
    def add(self, time, state):
        self.states.append(state)
        self.times.append(time)
        
    def clear(self,):
        self.states = []
        self.times = []
        
    def plot(self):
        plt.figure('States')
        plt.cla()
        self.states = np.array(self.states)
        self.times = np.array(self.times)
        for print_state, label, line_style in zip(self.print_list, self.plot_labels, self.line_styles):
            plt.plot(self.times, self.states[:,print_state], label = label, ls=line_style, lw=1)
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(1)
    
    def depth_plot(self):        
        plt.figure('3D map')
        self.states = np.array(self.states)
        self.times = np.array(self.times)
        xs = self.states[:,self.depth_plot_list[0]]
        ys = self.states[:,self.depth_plot_list[1]]
        zs = self.states[:,self.depth_plot_list[2]]
        t = self.times
        self.ax.scatter(xs,ys,zs,c=plt.cm.jet(t/max(t)))
        self.ax.plot3D(xs,ys,zs,linewidth=0.5)
        if len(self.depth_plot_list) > 3:
            t_xs = self.states[:,self.depth_plot_list[3]]
            t_ys = self.states[:,self.depth_plot_list[3]]
            t_zs = self.states[:,self.depth_plot_list[3]]
            self.ax.plot3D(t_xs,t_ys,t_zs,linewidth=1,color='r',ls='--')
        plt.grid(True)
        plt.draw()
        plt.pause(1)


class animation():
       
    def __init__(self,):
        current = visual.canvas.get_selected()
        current.delete()
        visual.rate(24)
        self.scene = visual.canvas(width=800, height=800)
        self.scene.autoscale = False
        self.scene.forward = visual.vector(-0.3,-1,-0.3)
        self.scene.up = visual.vector(0,0,1)
        self.scene.range = 3
        p_f = visual.vector(0,0,-BB_POS)
        p_w_l = visual.vector(-BB_POS,0,0)
        p_w_r = visual.vector(BB_POS,0,0)
        p_w_b = visual.vector(0,-BB_POS,0)        
        
        floor  = visual.box( pos=p_f, length = 2*BB_POS, height = 2*BB_POS,  width = 0.1)
        wall_1 = visual.box( pos=p_w_l, length = 0.1, height = 2*BB_POS,  width = 2*BB_POS)
        wall_2 = visual.box( pos=p_w_r, length = 0.1, height = 2*BB_POS,  width = 2*BB_POS)
        wall_3 = visual.box( pos=p_w_b, length = 2*BB_POS, height = 0.1,  width = 2*BB_POS)
        
        arm_1 = visual.box( length=2*D, height=0.05, width=0.05, color=visual.color.blue)
        arm_2 = visual.box( length=2*D, height=0.05, width=0.05, color=visual.color.red )
        arrow = visual.arrow( axis = visual.vector(0,0,0.3), shaftwidth=0.05)
        
        sphete = visual.sphere(radius=0.1, color = visual.color.red, opacity = 0.5)
        arm_2.rotate(angle=np.pi/2, axis = visual.vector(0,0,1))
        
        self.PIL_BB = (8, 110, 800+8, 800+110)
        
        self.drone = visual.compound([arm_1, arm_2, arrow], origin = visual.vector(0,0,0), make_trail=True, trail_color = visual.color.cyan)

        
    def animate(self,states):        
        states = np.array(states)
        
        s_i=7
        i=0
        in_pos = visual.vector(states[i,0],states[i,2],states[i,4])     
        self.drone.pos = in_pos
        self.scene.center = in_pos
        phi, theta, psi = states[i,6], states[i,7], states[i,8]
        #x rotation
        self.drone.rotate(phi, visual.vector(1,0,0))
        #y rotation
        self.drone.rotate(theta, visual.vector(0,1,0))
        #z rotation
        self.drone.rotate(psi, visual.vector(0,0,1))        
        phi_ant, theta_ant, psi_ant = phi, theta, psi
        im = PIL.ImageGrab.grab(self.PIL_BB)
        im.save("./animation/{:04d}.png".format(i), "png")
        for x,y,z,phi,theta,psi in zip(states[::s_i,0],states[::s_i,2],states[::s_i,4],states[::s_i,6],states[::s_i,7],states[::s_i,8]):   
             visual.rate(24)
             drone_pos = visual.vector(x,y,z)
             self.drone.pos = drone_pos
             #z rotation
             self.drone.rotate(-psi_ant, visual.vector(0,0,1))
             #y rotation
             self.drone.rotate(-theta_ant, visual.vector(0,1,0))
             #x rotation
             self.drone.rotate(-phi_ant, visual.vector(1,0,0))
             #x rotation
             self.drone.rotate(phi, visual.vector(1,0,0))
             #y rotation
             self.drone.rotate(theta, visual.vector(0,1,0))
             #z rotation
             self.drone.rotate(psi, visual.vector(0,0,1))
             phi_ant, theta_ant, psi_ant = phi, theta, psi

             self.scene.center = drone_pos
             # scene.forward = visual.vector(0,-1,0)
             i+=1
             self.scene.waitfor("draw_complete")
             im = PIL.ImageGrab.grab(self.PIL_BB)
             im.save("./animation/{:04d}.png".format(i), "png")
        input('Enter para parar o servidor')
        visual.no_notebook.stop_server()
