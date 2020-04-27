import gym
import numpy as np 
from gym import wrappers

env = gym.make('CartPole-v0')

#Initialise variables for the runs
longestRun = 0
episodeLengths = []
bestWeights = np.zeros(4)

for i in range(100):
    #Each run will have a set of weights, which when applied to the observation data (also a 4-tuple),
    #will return a real numbered result. If this is above zero, the cart moves right, if not, left. 
    new_weights = np.random.uniform(-1.0,1.0,4)
    length = []
    
    #Each set of weights gets 100 trials to form an average 'performance', measured by the count
    #of steps before failure. 
    for j in range(100):
        observation = env.reset()
        done = False
        count = 0
        while not done:
            #env.render()
            count+=1
            action = 1 if np.dot(observation,new_weights) > 0 else 0 #Dot product determines next action
            observation, reward, done, _ = env.step(action)
            
            if done:
                break
        length.append(count)

    #After 100 runs, the weights are evaluated, if they produce a longer average run length, they 
    #become the new 'best weights'.    
    average_length = np.mean(length)                                
    if average_length > longestRun:
        longestRun = average_length
        bestWeights = new_weights
    episodeLengths.append(average_length)

    if i % 10 == 0:
        print('Longest run: ', longestRun)


#Final rendered run using the best weights.
done = False
count = 0
#env = wrappers.Monitor(env, 'MovieFiles', force=True) #Include this if you want a saved file of render.
observation = env.reset()

while not done:
    env.render()
    count+=1
    action = 1 if np.dot(observation,bestWeights) > 0 else 0
    observation, reward, done, _ = env.step(action)
    
    if done:
        break

print("game lasted ", count, ' steps')