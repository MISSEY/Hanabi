import gym
import numpy as np
import time, pickle, os
import pandas as pd
import plotting
import matplotlib 
import matplotlib.style 
matplotlib.style.use('ggplot')
from IPython.display import clear_output

env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96
reward_save =[]
Q = np.zeros((env.observation_space.n, env.action_space.n))


episode_lengths = np.zeros(total_episodes)
episode_rewards = np.zeros(total_episodes)
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

rewards = 0
# Start

for episode in range(total_episodes):
    state = env.reset()
    t = 0
    rewards_save =[]
    while t < max_steps:
        clear_output(wait=True)
        #env.render()

        action = choose_action(state)
        #print("action space",env.action_space)
        #print(action)
        #print(state)
        

        state2, reward, done, info = env.step(action)
        
        rewards_save.append(reward)
        #print(np.array(rewards_save).shape)
        learn(state, state2, reward, action)

        state = state2

        t += 1
        
        episode_rewards[episode] += reward 
        episode_lengths[episode] = t
        
        rewards+=1
       
        if done:
            break

        #time.sleep(0.8)
    #print(episode)
    reward_save.append(np.sum(rewards_save)/np.array(rewards_save).shape[0])

print ("Score over time: ", rewards/total_episodes)
print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
    
print(episode_rewards)

