# Algoritmo Inicial de DDPG

Foi obtido por enquanto apenas controle de um sistema com todos os 12 graus de liberdade do quadrotor, com controle de posição no sistema de coordenadas inercial.
O algoritmo é de difícil convergencia, visto que são 12 graus de liberdade, subatuados e fortemente acoplados. A atuação do controlador se dá diretamente na velocidade angular do motor.
O modelo utilizado é de um drone aproximado aos DJI, com um peso mais elevado e constante de tempo maior, mas provavelmente se comportará bem para qualquer drone, dado um passo de tempo adequado.

Agradecimentos a Anuradha Weeraman em seu reddit: https://github.com/aweeraman/reinforcement-learning-continuous-control onde seu algoritmo de Atuação e Supervisão foi adaptado para utilização neste trabalho.
Agradecimentos a Yan Pan Lau em seu reddit: https://github.com/yanpanlau/DDPG-Keras-Torcs onde seu algoritmo também serviu de inspiração.
Agradecimentos a Dornal Byrne pela sua explicação sobre algoritmos DDPG TD3, sem o TD3 o sistema não convergiria. https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93

O Ruído gaussiano foi substituido novamente pelo ruído OU (Ornstein–Uhlenbeck) que é um ruído 'derivativo', adicionando ruído, média e variância em um sinal já existente.

