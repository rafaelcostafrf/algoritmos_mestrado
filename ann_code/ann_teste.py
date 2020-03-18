import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch

x_saida=[]
y_saida=[]


## DEFINICAO DA DINAMICA
def pos(t,x,u):
    m = 10
    return np.array([x[1], (u/m)])

ti, tf = 0, 5
passos = 100
t = np.linspace(ti,tf,passos)
t_in = t[0:len(t)-1]
t_fi = t[1:]

y = np.zeros((len(t), 2))
u = np.zeros((len(t), 1))



## DEFINICAO DA REDE NEURAL
#Q e a divisao entre database de treinamento e teste
Q = 0.75

# N tamanho da batelada; D_in dimensao da entrada;
# H e o tamanho dos neurons escondidos; D_out dimensao de saida.
N, D_in, H, D_out = 250, 7, 300, 2
x_nn = torch.randn(N, D_in)
y_nn = torch.randn(N, D_out)

T = int(N*Q)
R = N-T

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

y_inicial = np.random.rand(N,2)*10
entradas = np.random.rand(N,passos)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6
optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


## CALCULO DA DINAMICA E SEPARACAO DAS LISTAS DE DADOS
for y_0,u in zip(y_inicial,entradas):
    y[0,:] = y_0 
    for i,t_i,t_f in zip(range(0,passos),t_in,t_fi):
        r = integrate.solve_ivp(pos, (t_i, t_f), y[i], args=[u[i]])
        y[i+1]=r.y[:,-1]  
    atrasos = 3
    atraso_3 = y[0:len(y)-atrasos,:]
    atraso_2 = y[1:len(y)-atrasos+1,:]
    atraso_1 = y[2:len(y)-atrasos+2,:]
    atraso_0 = y[3:len(y)-atrasos+3,:]
    x_nn = np.zeros([len(atraso_0),D_in])
    y_nn = np.zeros([len(atraso_0),D_out])
    
    for i,pos_0,pos_1,pos_2,pos_3,entrada in zip(range(len(atraso_0)),atraso_0,atraso_1,atraso_2,atraso_3,u[3:len(y)]):
        x_nn[i,:]=[pos_1[0],pos_1[1],pos_2[0],pos_2[1],pos_3[0],pos_3[1],entrada]
        y_nn[i,:]=[pos_0[0],pos_0[1]]
        
    x_nn = torch.Tensor(x_nn)
    y_nn = torch.Tensor(y_nn)
    x_saida.append(x_nn)    
    y_saida.append(y_nn)

## NORMALIZACAO
divisor=0
for x,y in zip(x_saida,y_saida):
    max_x = torch.max(x)
    min_x = torch.min(x)
    max_y = torch.max(y)
    min_y = torch.min(y)
    divisor_i = max(abs(min_x),abs(min_y),max_x,max_y)
    if divisor_i>divisor:
        divisor=divisor_i

## TREINAMENTO DA REDE PARA CADA DADO GERADO
for k,x_nn, y_nn in zip(range(T),x_saida[0:T],y_saida[0:T]):
    for t in range(1000):
        y_pred = model(x_nn/divisor)
        loss = loss_fn(y_pred,y_nn/divisor)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        if t % 100 == 0:
            print("perca: %.10f batelada: %i" %(float(loss),k))

## TESTE COM O RESTO DA DATABASE
loss = 0
for k,x_nn, y_nn in zip(range(R),x_saida[T+1:N],y_saida[T+1:N]):
        y_pred = model(x_nn/divisor)
        loss += loss_fn(y_pred,y_nn/divisor)/R
print("%i TESTES -- perca: %.10f" %(T,float(loss)))
if (input("salvar resultado? digite s ou n: ")==("s")):
    str_in = input("Digite o nome do arquivo: ")
    str_final = ("modelos/"+str_in+".plk")
    torch.save(model.state_dict(), str_final)
    print(("Arquivo salvo em: " + str_final))


##PLOT DE UM TESTE PRA VISUALIZACAO

a = y_pred.detach().numpy()
b = y_nn.numpy()

plt.close('all')
plt.figure()
plt.title('posicao')
plt.plot(range(passos-3),a[:,0]*float(divisor))
plt.plot(range(passos-3),b[:,0])
plt.figure()
plt.title('velocidade')
plt.plot(range(passos-3),a[:,1]*float(divisor))
plt.plot(range(passos-3),b[:,1])
       