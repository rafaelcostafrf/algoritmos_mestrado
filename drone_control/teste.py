import numpy as np

class resultados:
    
    def __init__(self,x):
        self.entrada = np.array([x])
    
    def salva(self,entrada_2):
        self.entrada = np.append(self.entrada,np.array([entrada_2]),axis=0)
        
    def imprime(self,i):
        if i>=len(self.entrada):
            return
        print('Rodada: %i - %i Exploração %.2f Passos %i - %i Pontos %i' %(self.entrada[-1-i,0],self.entrada[-1,0],np.mean(self.entrada[-1-i:-1,1]),self.entrada[i,2],self.entrada[-1-i,2],np.mean(self.entrada[-1-i:-1,3])))
        plt.figure('Desenvolvimento - Erros')
        figura = plt.scatter(self.entrada[-1,0],np.mean(self.entrada[-1-i:-1,3]), c = 'blue', marker='x')
        plt.draw()
        plt.pause(0.1)
        
        
        