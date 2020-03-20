import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch

## DEFINICAO DA DINAMICA
def pos(t,x,u):
    m = 10
    return np.array([x[1], (u/m)])

## DEFINICAO DO INTEGRADOR NUMERICO
# ti - tempo inicial, tf - tempo final, 
# passos - passos de integracao, N número de simulacões a serem realizadas em uma batelada de treino
ti, tf = 0, 2
N, passos = 300, 50

## DEFINICAO DA REDE NEURAL
# D_in dimensao da entrada;
# H e o tamanho dos neurons escondidos; D_out dimensao de saida.
D_in, H, D_out = 7, 100, 2

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Tanh(),
    torch.nn.Linear(H,D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Q e a divisao entre database de treinamento e teste
Q = 0.75
T = int(passos*N*Q)
R = N*passos-T

## RECUPERA OS PESOS OBTIDOS DE OUTRO PROCESSO 
if (input("deseja carregar um modelo salvo? s ou n: ") == ("s")):
    model.load_state_dict(torch.load(("modelos/"+input("Digite o nome do arquivo: ")+".plk")))
    model.eval()

## VALORES INICIAIS E ENTRADAS ALEATORIAS (PARA O TREINAMENTO DA REDE)                          
y_inicial = np.random.rand(N,2)*2-np.ones([N,2])
entradas = np.random.rand(N,passos)*2-np.ones([N,passos])

## CALCULO DA DINAMICA E SEPARACAO DAS LISTAS DE DADOS

t = np.linspace(ti,tf,passos)
t_in = t[0:len(t)-1]
t_fi = t[1:]

y = np.zeros((len(t), 2))
u = np.zeros((len(t), 1))

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
    
    if 'x_saida' in locals():
        x_saida=np.append(x_saida,x_nn, axis=0)
        y_saida=np.append(y_saida,y_nn, axis=0)
    else:
        x_saida = x_nn
        y_saida = y_nn
        
x_saida = torch.Tensor(x_saida)
y_saida = torch.Tensor(y_saida)

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
x_train = x_saida[0:T]
y_train = y_saida[0:T]

indices = torch.randperm(len(x_train))    
x_train = x_train[indices]
y_train = y_train[indices]

for t in range(2000):    
    y_pred = model(x_train/divisor)
    loss = loss_fn(y_pred,y_train/divisor)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
    if t % 100 == 0:
        print("perca: %.10f progresso: %.2f%%" %(float(loss),t/2000*100))

## TESTE COM O RESTO DA DATABASE
x_test = x_saida[T+1:-1]
y_test = y_saida[T+1:-1]
loss = 0
y_pred = model(x_test/divisor)
loss += loss_fn(y_pred,y_test/divisor)/R
print("%i TESTES -- perca: %.10f" %(T,float(loss)))

##PLOT DE UM TESTE PRA VISUALIZACAO
a = y_pred.detach().numpy()
b = y_test.numpy()

plt.close('all')
plt.figure()
plt.title('posicao')
plt.plot(range(len(a)),a[:,0]*float(divisor))
plt.plot(range(len(a)),b[:,0])
plt.pause(1)
plt.figure()
plt.title('velocidade')
plt.plot(range(len(a)),a[:,1]*float(divisor))
plt.plot(range(len(a)),b[:,1])
plt.pause(1)

## SALVA OS PESOS OBTIDOS NO PROCESSO 
if (input("salvar resultado? digite s ou n: ")==("s")):
    str_in = input("Digite o nome do arquivo: ")
    str_final = ("modelos/"+str_in+".plk")
    torch.save(model.state_dict(), str_final)
    print(("Arquivo salvo em: " + str_final))