import gym
import numpy as np 
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

MAX_STATES = 10**4
GAMMA = 0.9 #Discount rate
ALPHA = 0.01 #Learning rate

#Takes a dictionary, returns the key-value pair with the highest value
def max_dict(d):
    max_v = float('-inf')
    for key, value in d.items():
        if value > max_v:
            max_v = value
            max_key = key
    return max_key, max_v

#Creates 10 bins per observation, based on their min/max value. Pole and cart velocity capped at
#-10 and 10 for practical purposes. 
def create_bins():
    bins = np.zeros((4,10))
    bins[0] = np.linspace(-4.8,4.8,10)
    bins[1] = np.linspace(-10,10,10)
    bins[2] = np.linspace(-.418,.418,10)
    bins[3] = np.linspace(-10,10,10)

    return bins

#Converts the continuous observation values to discrete values
def assign_bins(observation,bins):
    state = np.zeros(4)
    for i in range(4):
        #Uses indices of bins, which makes it easier to encode all states in the Q dictionary.
        state[i] = np.digitize(observation[i], bins[i]) 
    
    return tuple(state)

#Initialises the Q dictionary, which holds an expected reward for every state-action pair. 
def initialise_Q():
    q_dict = {}
    all_states = []

    #States are a 4-tuple observation, with 10 discrete bins.
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    all_states.append((i,j,k,l))
    
    for state in all_states:
        q_dict[state] = {}
        for action in range(env.action_space.n):
            q_dict[state][action] = 0 #Initial value for all state-action pairs

    return q_dict

def play_one_game(bins, q_dict, eps=0.5):
    observation = env.reset()
    done = False
    count = 0
    state = assign_bins(observation,bins)
    total_reward = 0

    while not done:
        count += 1
        
        #With probability 'epsilon', a random action is taken, to balance explore vs exploit. 
        if np.random.uniform() < eps:
            action = env.action_space.sample()
        else:
            action = max_dict(q_dict[state])[0]
        
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done and count < 200:
            reward = -300 #Penalise for having pole fall down

        new_state = assign_bins(observation,bins)
        a1, max_q_s1a1 = max_dict(q_dict[new_state])
        #QLearn updates the expected reward of the state-action pair by looking at the
        #generated reward and the expected reward of the new state.
        q_dict[state][action] += ALPHA*(reward + GAMMA*(max_q_s1a1) - q_dict[state][action])
        
        state = new_state
    
    return total_reward, count

def play_n_games(bins, n=10000):
    q_dict = initialise_Q()
    length = []
    reward = []

    for i in range(n):
        epsilon = 1 / np.sqrt(i+1) #Epsilon decreases over episodes, going from exploration to exploitation
        ep_reward, ep_length = play_one_game(bins,q_dict,epsilon)

        if i % 100 == 0:
            print(i, ": ", ep_reward)
        length.append(ep_length)
        reward.append(ep_reward)
    
    return length, reward

def plot_running_avg(total_rewards):
    n = len(total_rewards)
    running_avg = np.empty(n)
    for i in range(n):
        running_avg[i] = np.mean(total_rewards[max(0,i-100) : i+1]) #Averages last 100 episodes
    
    plt.plot(running_avg)
    plt.title("Running average reward")
    plt.show()

if __name__ == '__main__':
    bins = create_bins()
    ep_lengths, ep_rewards = play_n_games(bins)
    plot_running_avg(ep_rewards)

        