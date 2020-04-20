import datetime
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_fromfile():
    """"
        Reads best reward from file
    """
    File_reward = open('saves/best_reward.txt','r')
    v = float(File_reward.read())
    File_reward.close()
    File_solved = open('saves/best_solved.txt','r')
    s = float(File_solved.read())
    File_solved.close()
    return v, s

def write_tofile(T,epi_num,reward,solved,best=0):
    """"
        Writes to a txt file important data of the training process
        Inputs:
            T (int): Number of total steps
            epi_num (int): Number of episodes
            reward(float): Average Reward
            best(bool): if best, writes the usual line and overwrites best average on another txt file

    """
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M")
    line_values = str(' T: %i Ep. Num: %i Reward: %.4f Solved: %i           \n' %(T,epi_num,reward,solved))
    line = date+line_values
    File = open('saves/updates.txt', 'a')
    File.write(line)
    File.close()
    if best:
        File_reward = open('saves/best_reward.txt','w')
        File_reward.write(str(reward))
        File_reward.close()
        File_solved = open('saves/best_solved.txt','w')
        File_solved.write(str(solved))
        File_solved.close()



def rand_input(len_batch,noise_v,in_prob,numpy=0):
    """"
        Generates a random predetermined input from a set of actions.
        If the sum of probability in in_prob is lower than 1, the algorithm generates
        a random normal noise input.
        Inputs:
            len_bath (int): batch size
            noise_v (int): noise variance
            in_prob(4 size array float): array of probabilities of each action
            numpy(bool): if numpy, returns the noise converted to numpy, else return the noise converted to torch
    """
    noise=np.zeros([len_batch,4])
    in_up = np.array([1,1,1,1])
    in_x_roll = np.array([1,0,-1,0])
    in_y_roll = np.array([0,1,0,-1])
    in_z_roll = np.array([1,-1,1,-1])
    for i, _ in enumerate(noise):
        prob = np.random.random()
        noise_n = np.random.randn()*noise_v
        if prob < in_prob[0]:
            noise[i,:]=in_up*noise_n
        elif prob < in_prob[1]:
            noise[i,:]=in_x_roll*noise_n
        elif prob < in_prob[2]:
            noise[i,:]=in_y_roll*noise_n
        elif prob < in_prob[3]:
            noise[i,:]=in_z_roll*noise_n
        else:
            noise[i,:]=np.random.randn(4)*noise_v
    if numpy==0:
        return torch.from_numpy(noise).float().to(device)
    else:
        return noise[0]

class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, max_size=1e6):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, s_, a, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)

