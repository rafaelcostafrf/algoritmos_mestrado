import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint

global u
global m

def func(t,x):
    return torch.Tensor([x[1], (u[i]/m[i])])

## DEFINICAO DO INTEGRADOR NUMERICO
    
# t_i - tempo inicial, t_f - tempo final, 
# N número de simulacões a serem realizadas em uma batelada de treino, passos - passos de integracao 
t_i, t_f = 0, 30
N, passos = 100, 100*t_f


## DEFINICAO DA REDE NEURAL

# D_in dimensao da entrada;
# H e o tamanho dos neurons escondidos; D_out dimensao de saida.
D_in, H, D_out = 7, 60, 2


## INICIALIZACAO DA REDE NEURAL

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

if (input("deseja carregar um modelo salvo? s ou n: ") == ("s")):
    model.load_state_dict(torch.load(("modelos/"+input("Digite o nome do arquivo: ")+".plk")))
    model.eval()

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-2
optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


## INICIALIZACAO DO INTEGRADOR NUMERICO

t = torch.linspace(t_i,t_f,passos+1)
y0 = torch.Tensor([10,0])
T = torch.transpose(torch.stack((t[0:-1],t[1:])),0,1)
u = torch.ones(passos+1,1)
m = torch.ones(passos+1,1)

## INTEGRADOR NUMERICO

y=torch.zeros([passos+1,2])
y[0,:]=y0

divisor=15
x_train = torch.cat((y[-1:,:],y[-2:-1,:],y[-3:-2,:],u[-1:,:]),1)
y_train = y[-1:,:]
u[0] = model(x_train/divisor)[0][0]*0
u[1] = model(x_train/divisor)[0][1]*0

loss = loss_fn(torch.zeros(2),y[0,:])
loss.requires_grad=True
optmizer.zero_grad()
loss.backward(retain_graph=True)
optmizer.step()

for i, t_int in zip(range(passos),T):
    a = odeint(func,y[i,:],t_int)
    y[i+1,:] = a[1,:]
    if i>1 and i%2==0:
        x_train = torch.cat((y[i,:],y[i-1,:],y[i-2,:],u[i,:]),0)
        divisor = 1
        y_pred = model(x_train/divisor)
        u[i+1] = y_pred[0]
        u[i+2] = y_pred[1]
        loss = loss_fn(torch.Tensor([0,0]),torch.Tensor([y[i,0],y[i+1,0]]))
        loss.requires_grad=True
        optmizer.zero_grad()
        loss.backward(retain_graph=True)
        optmizer.step()
        if i % 100 == 0:
             print("perca: %.10f progresso: %.2f%%" %(float(loss),i/passos*100))
             
if (input("salvar resultado? digite s ou n: ")==("s")):
    str_in = input("Digite o nome do arquivo: ")
    str_final = ("modelos/"+str_in+".plk")
    torch.save(model.state_dict(), str_final)
    print(("Arquivo salvo em: " + str_final))
    
    
a = y.numpy()
t = t.numpy()

plt.close('all')
plt.figure()
plt.title('posicao')
plt.plot(t,a)
plt.pause(0.05)

