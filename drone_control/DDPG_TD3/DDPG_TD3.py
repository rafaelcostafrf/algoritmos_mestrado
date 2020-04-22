import sys
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt


#PACOTES DO MODELO
from DRONE_MODELO import DinamicaDrone, Q2Euler
from UTILITIES import read_fromfile, write_tofile, rand_input, ReplayBuffer
from NN_MODELS import Actor, Critic


P_DEBUG = 0

#TAXA DE APRENDIZADO DO CRITICO, DO ATOR, TAXA DE TRANSFERENCIA PARA A REDE ALVO E PESO DAS ACOES FUTURAS
LEARNING_RATE_CRITIC = 0.001
LEARNING_RATE_ACTOR = 0.001
TAU = 0.005
GAMMA = 0.99

#SEMENTE ALEATORIA PARA REPRODUTIBILIDADE
SEED = 0

#NUMERO DE OBSERVACOES ANTES DE COMECAR O TREINAMENTO
OBSERVATION = 10000

#NUMERO DE OBSERVACOES COM A POLITICA ATUAL DO MODELO SALVO (ANTES DO TREINAMENTO)
EXP_BEFORE_TRAIN = 10000

#NUMERO TOTAL DE PASSOS DE EXPLORAÇÃO
EXPLORATION = 10000000

#MEDIA DE RECOMPENSA (APENAS PARA O PRINT NA TELA)
N_MEAN = 10

#TAMANHO DA BATELADA DE TREINAMENTO (PARA CADA PASSO DE TEMPO EM TRAIN FREQUENCY)
BATCH_SIZE = 100
TRAIN_FREQUENCY = 1

#FREQUENCIA DE ATUALIZACAO DA POLITICA (ATOR)
POLICY_FREQUENCY = 2

#RUÍDO DE EXPLORACÃO E RUIDO NA POLITICA NO TREINAMENTO
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = .6

#TAXA DE DECAIMENTO EXPONENCIAL DO RUIDO
EXPLORATION_NOISE_DECAY = 0

#CLIP DE RUIDO (SE NECESSARIO) E PROBABILIDADE DE ENTRADAS PRE DETERMINADAS
PROB_Z = 0
PROB_PHI = 0
PROB_THETA = 0
PROB_PSI = 0
IN_PROB=np.array([PROB_Z,
                  PROB_Z+PROB_PHI,
                  PROB_Z+PROB_PHI+PROB_THETA,
                  PROB_Z+PROB_PHI+PROB_THETA+PROB_PSI])

#RECOMPENSA MINIMA PARA FINALIZAR TREINAMENTO
REWARD_THRESH = np.inf

#PASSO DA SIMULACAO, PASSOS MAXIMOS DO SISTEMA E FREQUENCIA DE VALIDACAO
TIME_STEP = 0.01
MAX_ENV_STEPS = 1000
EVAL_FREQUENCY = 20000

T = 5

#ALGORITMO TD3 BASEADO EM https://gist.github.com/djbyrne/d58aa1abfe0a14e68686b2c514120d49#file-td3-ipynb
#CORRECOES IMPORTANTES FORAM FEITAS, TAMBÉM FORAM FEITAS ADAPTACOES PARA O AMBIENTE DO QUADROTOR
class TD3(object):
    """Agent class that handles the training of the networks and provides outputs as actions

        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use

    """

    def __init__(self, state_dim, action_dim, max_action, env):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE_CRITIC)

        self.max_action = max_action
        self.env = env

        try:
            self.load()
            print('Past models loaded')
        except:
             print('Past models could not be loaded')


    def select_action(self, state, noise):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons

            Returns:
                action (float): action clipped within action range

        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + rand_input(1, noise, IN_PROB, numpy=1))

        return action.clip(MIN_ACTION, MAX_ACTION)


    def train(self, replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):
        """Train and update actor and critic networks

            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): training iteration number
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update from main networks to target networks

            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network

        """

        # for it in range(iterations):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn(action.size())*policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(MIN_ACTION, MAX_ACTION)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q)+F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Delayed policy updates
        if iterations % policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))


    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename),map_location=device))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename),map_location=device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename),map_location=device))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename),map_location=device))

class Runner():
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env, agent, replay_buffer):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.inicial()
        self.done = False

    def next_step(self, episode_timesteps, noise):

        action = self.agent.select_action(np.array(self.obs), noise)

        # Perform action
        new_obs, reward, done = self.env.passo(action)
        done_bool = float(done)

        # Store data in replay buffer
        self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

        self.obs = new_obs

        if done:
            self.obs = self.env.inicial()
            done = False
            return reward, True

        return reward, done

def evaluate_policy(policy, env, eval_episodes=10):

    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations
            var_reward (float): standard deviation over the number of evaluations
    """
    with torch.no_grad():
        states = []
        actions = []
        rewards = []
        n_solved = 0
        eval_steps = 0
        for i in range(eval_episodes):
            reward_i = 0
            obs = env.inicial()
            done = False
            while not done:
                action = policy.select_action(np.array(obs), noise=0)
                if i == eval_episodes-1:
                    q = np.array([env.y_0[6:10]]).T
                    phi,theta,psi = Q2Euler(q)
                    ang = np.array([phi,theta,np.sin(psi/2)])
                    state_conv = np.concatenate((env.y_0[0:6], ang))
                    states.append(state_conv)
                    actions.append(action)
                obs, reward, done = env.passo(action)
                resolvido = env.resolvido
                reward_i += reward
                eval_steps += 1
                if resolvido:
                    n_solved += env.resolvido
                    env.reset=True
                    break
            rewards.append(reward_i)
        avg_reward = np.mean(rewards)
        var_reward = (np.var(rewards))**(0.5)
        states = np.array(states)
        actions = np.array(actions)
        t = np.arange(0, len(states), 1)*TIME_STEP

        plt.cla()

        plot_list = [0,2,4,6,7,8]
        plot_label = ['x', 'y', 'z', 'phi', 'theta', 'psi']
        line_style = ['-', '-', '-', '--', '--', '--']
        plot_f = ['f1', 'f2', 'f3', 'f4']
        for a,lab,ls in zip(plot_list, plot_label, line_style):
            plt.plot(t, states[:,a], label =lab, ls=ls, lw=1)
        for a,lab in zip(range(4), plot_f):
            plt.plot(t, actions[:,a], label=lab, ls=':', lw=1)
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.title(avg_reward)
        plt.pause(1)
    return avg_reward, var_reward, n_solved/eval_episodes

def observe(env,replay_buffer,policy_env=0):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for
            polivy_env (bool): if observe is meant to run on random actions or on saved best policy

    """
    if policy_env == 0:
        observation_steps = OBSERVATION
    else:
        observation_steps = EXP_BEFORE_TRAIN
    time_steps = 0
    obs = env.inicial()
    done = False

    while time_steps < observation_steps:
        if policy_env == 0:
            action = np.random.normal(0,MAX_ACTION/2,[4])
            action = np.clip(action,MIN_ACTION,MAX_ACTION)
        else:
            action = policy_env.select_action(np.array(obs), EXPLORATION_NOISE)
        new_obs, reward, done = env.passo(action)

        replay_buffer.add((obs, new_obs, action, reward, done))
        time_steps += 1

        obs = new_obs

        if done:
            obs = env.inicial()
            done = False
        if policy_env == 0:
            print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
            sys.stdout.flush()
        else:
            print("\rPopulating Buffer with policy actions {}/{}.".format(time_steps, observation_steps), end="")
            sys.stdout.flush()
    print('')


def train(agent, test_env, val_env):
    """Train the agent for exploration steps

        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run

    """
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    train_timesteps = 0
    train_iteration = 0
    eval_episodes = 10
    eval_episodes_check = 100
    done = False
    evaluations = []
    best_avg = BEST_AVG
    best_n_solved = BEST_SOLVE
    noise = EXPLORATION_NOISE

    while total_timesteps < EXPLORATION:
        if total_timesteps != 0:
            if done:
                #Evaluate episode
                if timesteps_since_eval >= EVAL_FREQUENCY:
                      timesteps_since_eval %= EVAL_FREQUENCY
                      eval_reward, var_reward, n_solved = evaluate_policy(agent, val_env, eval_episodes)
                      evaluations.append(eval_reward)
                      print("\rTotal T: {:d} Episode Num: {:d} Total Reward: {:.2f}±{:.2f} Avg Reward: {:.2f} Exploration: {:.2f} Solved: {:.3f} \n".format(
                        total_timesteps, episode_num, eval_reward,var_reward, np.mean(evaluations[-N_MEAN:]), noise, n_solved), end="")
                      sys.stdout.flush()
                      if eval_reward > best_avg*0.85 or n_solved > best_n_solved*0.85:
                         #IF PRELIMINAR IS BETTER, EVALUATE AGAIN WITH MORE EPISODES#
                         eval_reward, var_reward, n_solved = evaluate_policy(agent, val_env, eval_episodes_check) 
                         print("\r ------ SECOND EVALUATION ----- Total Reward: {:.2f}±{:.2f} Avg Reward: {:.2f} Exploration: {:.2f} Solved: {:.3f} \n".format(
                             eval_reward,var_reward, np.mean(evaluations[-N_MEAN:]), noise, n_solved), end="")
                         if eval_reward > best_avg or n_solved > best_n_solved:
                             if n_solved:
                                 best_avg = 1e10
                             else:
                                 best_avg = eval_reward
                             best_n_solved = n_solved
                             write_tofile(total_timesteps,episode_num,best_avg,n_solved,1)
    
                             print("\n------------------\nSaving best model\n------------------\n")
                             agent.save("best_avg","saves")
                             if n_solved >= 0.95:
                                 print('\n------------------------------------------------\n------------------------------------------------\n------------------------------------------------\nModel training finished, objective reward over evaluation accomplished.')
                                 break
                      else:
                         write_tofile(total_timesteps,episode_num,eval_reward,n_solved)

                noise = EXPLORATION_NOISE*np.exp(-episode_num*EXPLORATION_NOISE_DECAY)
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        if train_timesteps > TRAIN_FREQUENCY:
            train_iteration+=1
            print('\rTraining: %.2f%%          ' %(timesteps_since_eval/EVAL_FREQUENCY*100),end='')
            train_timesteps = total_timesteps % TRAIN_FREQUENCY
            agent.train(REPLAY_BUFFER, train_iteration, BATCH_SIZE, GAMMA, TAU, POLICY_NOISE, NOISE_CLIP, POLICY_FREQUENCY)
        reward, done = RUNNER.next_step(episode_timesteps, noise)
        episode_reward += reward
        train_timesteps += 1
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1



ENV = DinamicaDrone(TIME_STEP, MAX_ENV_STEPS, T, P_DEBUG)
EVAL_ENV = DinamicaDrone(TIME_STEP, MAX_ENV_STEPS, T, P_DEBUG)
ACTION_DIM = 4
STATE_DIM = (ENV.estados+ACTION_DIM)*T
MIN_ACTION = -1
MAX_ACTION = 1

# Set seeds
ENV.seed(SEED)
EVAL_ENV.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set default Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Opens a plot window maximized
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

#Load best values from past models
try:
    BEST_AVG, BEST_SOLVE = read_fromfile()
    print('Best Reward Loaded: %.2f Best Solve Loaded: %.3f' %(BEST_AVG,BEST_SOLVE))
except:
    EXP_BEFORE_TRAIN=0
    BEST_AVG = -np.inf
    BEST_SOLVE = 0
    print('Could not load best reward and solution, best reward and solution reset')


POLICY = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION, ENV)
REPLAY_BUFFER = ReplayBuffer()
RUNNER = Runner(ENV, POLICY, REPLAY_BUFFER)
observe(ENV, REPLAY_BUFFER)
observe(ENV, REPLAY_BUFFER, policy_env=POLICY)
train(POLICY, ENV, EVAL_ENV)
