# Algoritmo Inicial de DDPG

Foi obtido por enquanto apenas controle de um sistema com um grau de liberdade em Z.
O algoritmo já converge a partir de 1000 iterações, mas demonstra ainda possível melhora até 3000 iterações (controle mais suave e preciso.)
Nos próximos commits irei adicionar os outros estados do sistema (pitch yaw e roll).


Agradecimentos a Anuradha Weeraman em seu reddit: https://github.com/aweeraman/reinforcement-learning-continuous-control onde seu algoritmo de Atuação e Supervisão foi adaptado para utilização neste trabalho.

Uma das adaptações inseridas foi a utilização de ruído gaussiano, mais simples, e decaimento exponencial da chance de ocorrer uma ação aleatória, melhorando muito na convergencia deste problema.
