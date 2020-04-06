# Algoritmo Inicial de DDPG

Foi obtido por enquanto apenas controle de um sistema com todos os 12 graus de liberdade do quadrotor, com controle de posição no sistema de coordenadas inercial.

O algoritmo é de difícil convergencia, por volta de 5000 épocas, com máximo de 500 passos por época, demorou cerca de 12 horas em um computador 'mediano' utilizando processamento CUDA, sendo necessário trocar a taxa de aprendizado algumas vezes durante o processo.

Cada época é iniciada com velocidades e posições iniciais distribuídas aleatoriamente entre [-0.5,0.5]. O bounding box da simulação é [-10,10] na posição linear, [-10,10] nas velocidades lineares e [-pi,pi] nas posições angulares, se o sistema passar desses pontos, a simulação é resetada.

A convergência é difícil pelo fato de que são 12 graus de liberdade, subatuados e fortemente acoplados. A atuação do controlador se dá diretamente na velocidade angular do motor.

O modelo utilizado é de um drone aproximado aos DJI, com um peso mais elevado e constante de tempo maior, mas provavelmente se comportará bem para qualquer drone, dado um passo de tempo adequado.

Agradecimentos a Anuradha Weeraman em seu reddit: https://github.com/aweeraman/reinforcement-learning-continuous-control onde seu algoritmo de Atuação e Supervisão foi adaptado para utilização neste trabalho.

Agradecimentos a Yan Pan Lau em seu reddit: https://github.com/yanpanlau/DDPG-Keras-Torcs onde seu algoritmo também serviu de inspiração.

Agradecimentos a Dornal Byrne pela sua explicação sobre algoritmos DDPG TD3, sem o TD3 o sistema não convergiria. https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93

O Ruído gaussiano foi substituido novamente pelo ruído OU (Ornstein–Uhlenbeck) que é um ruído 'derivativo', adicionando ruído, média e variância em um sinal já existente.

Ainda é necessário fazer uma limpeza no código, principalmente para melhor entendimento e talvez uma melhoria no desempenho, talvez rodar épocas em paralelo(?).

![Resultados](/modelos/resultado_1.png)

