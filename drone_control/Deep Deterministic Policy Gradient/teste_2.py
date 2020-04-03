from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt 
from collections import deque
import time

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
                       
        return np.transpose(np.dot(R_z,np.dot(R_y,R_z))),np.linalg.inv(W_R)


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
    
                              
    return np.array([vel_x,accel_x,vel_y,accel_y,vel_z,accel_z,vel_phi,vel_theta,vel_psi,accel_w_xx,accel_w_yy,accel_w_zz])

y_0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
passo_t = 0.01
entrada = np.array([[0.56,0.57143,0.58,0.57143]])
a = integrate.solve_ivp(eq_drone,(0,passo_t),y_0, args=[entrada])
for i in range(1000):
    y_0 = a.y[:,-1]
    a = integrate.solve_ivp(eq_drone,(0,passo_t),y_0, args=[entrada])
    print(a.y)