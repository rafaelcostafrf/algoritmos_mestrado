import numpy as np
from scipy import integrate


def eq_drone(t,x,u):
    m = 1                   #Massa em KG
    g = -10               #Constante Gravitacional
    return np.array([x[1], (u/m+g)])
    
    
def din_drone(ti,anterior,controle):
    
    #funcao din_drone - Realimentando a dinamica da saida na entrada faz o sistema "andar pra frente" no tempo passo_t
    #esta funcao sera utilizada no processo de deep learning do drone com controles discretos (aumenta rpm do motor, ou diminui).
    #por enquanto a dinamica esta sendo simulada apenas no eixo Z, para fins de testes de viabilidade
    #a dinamica também dá um sinal de parar, e um sinal de score, se estiver dentro de uma faixa determinada de valores
    #somando 1 para cada passo de tempo dentro dos parametros, ou reset se sair fora da caixa (bounding box)
    
    b_b_min = -0.1          #bouding box minimo - metros
    b_b_max = 100           #bouding box maximo - metros

    
    passo_t = 0.01          #passo em segundos
    passo_w = 10            #passo da RPM
    B = 10                  #constante do motor
    
    reset = anterior[5]
    score = 0 if reset == True else anterior[6]
    
    if reset == False:
        y_0 = anterior[2:4]
        w_0 = anterior[4]
    else:
        y_0 = np.random.rand(2)*3
        w_0 = 0
        score = 0
        
    
    w = w_0 + passo_w  if (controle==[1,0] or controle ==[0,1]) else w_0       #definicao da RPM a partir do sinal de controle              
    entrada = B*w**2 if controle==[1,0] else -B*w**2

    y = (integrate.solve_ivp(eq_drone,(ti,ti+passo_t),y_0, args=[entrada])).y  #integrando a dinâmica
    
    reset = False if (y[-1,0]<b_b_max and y[-1,0]>b_b_min) else True
    score = score + 1 if reset == False else score
    
    return [y[0,0],y[-1,0],y[0,1],y[-1,1],w,reset,score]

a = [0,0,0,0,0,True,0]  #inicialização da entradas, o importante é a posição 5 estar em RESET (True)

for i in range(100):    
    a=din_drone(0,a,[1,0])


            