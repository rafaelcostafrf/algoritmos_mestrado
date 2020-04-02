from collections import deque
import pickle

a = deque(maxlen=10)

a = [1,1,1,1,1,1,1,1,1,1]

with open('teste','wb') as teste_file:
    pickle.dump(a,teste_file)
    
with open('teste','rb') as teste_file:
    b = pickle.load(teste_file)
    
    
    

    
    
    
    
    
