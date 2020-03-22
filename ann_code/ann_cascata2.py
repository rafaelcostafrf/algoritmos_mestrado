import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
from torch.autograd import Variable

global u
global m

def func(t,x):
        return torch.Tensor([x[1], (u[i]/m[i])])
    
def func_2(t,x):
        return torch.Tensor([x[1], (u_s[j]/m[i])])

def f_perca(y_entrada):
    y1 = y_entrada[:,0:2]
    y0 = y_entrada[:,2:4]
    u0 = y_entrada[:,4]
    u_cte = 0
    v_cte = 0.3
    val = ((y1[:,0]**2)+(y0[:,0]**2)+u_cte*(u0**2)+v_cte*((y1[:,1]**2)+(y0[:,1]**2))).mean()
    return val

def init_weights(m):
    m.bias.data.fill_(0)


## DEFINICAO DO INTEGRADOR NUMERICO
    
# t_i - tempo inicial, t_f - tempo final, 
# N número de simulacões a serem realizadas em uma batelada de treino, passos - passos de integracao 
t_i, t_f = 0, 120
N, passos, H_pass = 100, 10*t_f, 1


## DEFINICAO DA REDE NEURAL

# D_in dimensao da entrada;
# H e o tamanho dos neurons escondidos; D_out dimensao de saida.
D_in, H, D_out = 4, 200, 1

## INICIALIZACAO DA REDE NEURAL

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)
model.apply(init_weights)

# if (input("deseja carregar um modelo salvo? s ou n: ") == ("s")):
#     model.load_state_dict(torch.load(("modelos/"+input("Digite o nome do arquivo: ")+".plk")))
#     model.eval()

learning_rate = 0.005
optmizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


## INICIALIZACAO DO INTEGRADOR NUMERICO

t = torch.linspace(t_i,t_f,passos+1)
T = torch.transpose(torch.stack((t[0:-1],t[1:])),0,1)
u = torch.zeros(passos+1,1)
m = torch.ones(passos+1,1)*1

## INTEGRADOR NUMERICO

y=torch.zeros([passos+1,2])
y[0,:]=torch.Tensor([1,0])
divisor=1


for i, t_int in zip(range(passos),T):
    a = odeint(func,y[i,:],t_int)
    y[i+1,:] = a[1,:]
    if i>H_pass+1:
        for k in range(1):
            x_train = torch.empty(H_pass,D_in)
            y_train = torch.empty(H_pass,5)
            y_in = torch.empty(H_pass,2)
            for j in range(H_pass):
                x_train[j,:] = torch.cat((y[i-H_pass+j,:],y[i-1-H_pass+j,:]),0)
            u_s = model(x_train)  
            for j in range(H_pass):
                y_train[j,:] = torch.cat((y[i-H_pass+j,:],y[i-H_pass+j+1,:],u_s[j]),0)
            loss = f_perca(y_train)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
            if k % 10 == 0 and i % 100 == 0:
                print("perca: %.2f progresso: %.2f%%" %(float(loss),i/passos*100))
        x_saida = torch.cat((y[i+1,:],y[i,:]), 0)        
        u[i+1]= model(x_saida)
    

             
# if (input("salvar resultado? digite s ou n: ")==("s")):
#     str_in = input("Digite o nome do arquivo: ")
#     str_final = ("modelos/"+str_in+".plk")
#     torch.save(model.state_dict(), str_final)
#     print(("Arquivo salvo em: " + str_final))
    
    
a = y.numpy()
t = t.numpy()

plt.close('all')
plt.figure()
plt.title('posicao')
plt.plot(t,a)
plt.pause(0.05)

