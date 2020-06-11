
import numpy as np

import tensorflow as tf
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML
import random
from IPython import display as ipythondisplay
from IPython.display import clear_output


from hanabi_learning_environment import rl_env

    
class Qnetwork(tf.keras.Model):
    def __init__(self, hidden_units, num_actions,num_states):
        super(Qnetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                units, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation = 'softmax')

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output



class DDQNAgent():
    """Agent using Double DQN """
    def __init__(self, config,hidden_units,tau,gamma,batch_size,max_experiences,min_experiences,lr):
        """Initialize the agent."""
        self.config = config
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        
        self.env = config['env']
        
        self.train_net = Qnetwork(hidden_units,self.action_size,self.state_size)
        #self.train_net.build((1,self.state_size))
        self.target_net = Qnetwork(hidden_units,self.action_size,self.state_size)
        #self.target_net.build((1,self.state_size))
        self.lr = lr
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.tau = tau
        
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        
        self.experiences= []

    
    def get_prob_action(self,legal_moves,action_dist, state, eps):
        if random.random() < eps:
            action = np.random.choice(legal_moves)
            #actions = [x if i in legal_moves else 0 for i, x in enumerate(action_dist)]
            #if sum(actions)!=0:
                #actions = [x/sum(actions) for x in actions]
                #action = np.random.choice(list(range(self.action_size)), p=actions)
            #else:
                #action = None
        else:
            logits = np.array(self.train_net(np.array(state).reshape(1, -1))[0])
            actions = [x if i in legal_moves else 0 for i, x in enumerate(logits)]
            action = np.argmax(actions)
        
        return action

        
    def act(self, observation):
        if observation['current_player_offset'] != 0:
            return None
        if len(self.experiences) < self.min_experiences:
            return 0
        
        ids = np.random.randint(low=0, high=len(self.experiences), size=self.batch_size)
        states = np.array([np.array(self.experiences[id_][0]['vectorized']).astype(np.float32) for id_ in ids])
        actions = np.array([self.experiences[id_][1] for id_ in ids])
        rewards = np.array([self.experiences[id_][2] for id_ in ids])
        next_states = np.array([(np.zeros(self.state_size)if self.experiences[id_][3] is None else np.array(self.experiences[id_][3]['vectorized']).astype(np.float32)) for id_ in ids])

        #break

        # Obtaining Q values for current state
        
        Q_train= self.train_net(states)
        # Obtaining Q values for next state
        Q_train_prime = self.train_net(next_states)
        

        # target q , will be update later
        Q_train_t = Q_train.numpy()

        updates = rewards.astype(np.float32)

        #ids those next states are non zero, those ids are only valid one
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(self.batch_size)
        
        # Q value Update equation
        A_prime = np.argmax(Q_train_prime.numpy(), axis=1)
        Q_target = self.target_net(next_states)
        updates[valid_idxs] += self.gamma * Q_target.numpy()[batch_idxs[valid_idxs], A_prime[valid_idxs]]
        Q_train_t[batch_idxs, actions] = updates

        with tf.GradientTape() as tape:
            logits = self.train_net(states)
            action_dist = logits.numpy()
            #legal_moves = observation['legal_moves_as_int']
            #action_r = self.get_prob_action(legal_moves, action_dist[0],states,eps)
            
            loss = self.mse(Q_train_t, logits)
            
            gradients = tape.gradient(loss, self.train_net.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.train_net.trainable_variables))

        #copying train network into target network partially 
        for t, e in zip(self.target_net.trainable_variables, self.train_net.trainable_variables):
            t.assign(t * (1 - self.tau) + e * self.tau)
        
        return action_dist
    
    def add_experiences(self, exp):
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)

AGENT_CLASSES = {"DDQNAgent": DDQNAgent}

class Runner(object):
    """Runner class"""
    
    def __init__(self, flags, max_epsilon=1,min_epsilon = 0.00,lambda_ = 0.00005,
    gamma = 0.95, batch_size = 32, tau=0.08, max_experiences=400000,
    min_experiences = 96,hidden_units =[30,30], lr =0.00001):
        """Initialize runner"""
        self.flags = flags
        self.environment = rl_env.make('Hanabi-Very-Small', num_players=self.flags['players'])
        self.agent_config = {"players": flags['players'],
                             'state_size':self.environment.vectorized_observation_shape()[0],
                             'action_size': self.environment.num_moves(),
                             'env': self.environment}
        self.agent_class = AGENT_CLASSES[flags['agent_class']]
        
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.lambda_ = lambda_
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        self.hidden_units = hidden_units
        self.lr = lr
        
    def run(self):
        """Run episodes"""
        rewards = []
        agents = [self.agent_class(self.agent_config,self.hidden_units,self.tau,self.gamma,self.batch_size,self.max_experiences,
    self.min_experiences,self.lr) 
                  for _ in range(self.flags['players'])]
        steps = 0
        eps = self.max_epsilon
        for episode in range(self.flags['num_episodes']):
            print("episode: ",episode)
            observations = self.environment.reset()
            done = False
            episode_reward = 0
            while not done:
                for agent_id, agent in enumerate(agents):
                    observation = observations['player_observations'][agent_id]
                    action_dist = agent.act(observation)
                    legal_moves = observation['legal_moves_as_int']
                    if action_dist is not None:
                        if(type(action_dist) == type(0)):
                            action = np.random.choice(legal_moves)
                        else:
                            action = agent.get_prob_action(legal_moves, action_dist[0],observation['vectorized'],eps)
                    else:
                        action = None
                    #print(action)
                    #if(flag_action==0):
                        #action = self.environment.game.get_move_uid((self.environment._build_move(action)))
                    #print(self.environment._build_move(action))
                    #a = self.environment.game.get_move_uid((self.environment._build_move(action)))
                    #print(a)
                    #print(self.environment.game.get_move(a).to_dict())
                    #print("action")
                    # convert action from uid to dict
                    #if action is not None:
                        #action = self.environment.game.get_move(action).to_dict()
                    if observation['current_player'] == agent_id:
                        assert action is not None
                        current_player_action = action
                        break
                    else:
                        assert action is None
                        
                # Make an envirnment step:
                print("Action number :",current_player_action)
                print('Agent: {} action: {}'.format(observation['current_player'],self.environment.game.get_move(current_player_action).to_dict()))

                next_observations, reward, done, _ = self.environment.step(self.environment.game.get_move(current_player_action).to_dict())
                
                episode_reward+=reward
                
                # store episode memory for the agent
                for agent_id, agent in enumerate(agents):
                    agent.add_experiences((observations['player_observations'][agent_id],action,reward,next_observations['player_observations'][agent_id]))
                observations = next_observations
                eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lambda_ * steps)
                steps +=1 
            rewards.append(episode_reward)
            
            if episode % 50 == 0:
                logging.info("Episode: {}, Max reward:{}, Avg Reward:{}"
                .format(episode, max(rewards[-50:]), sum(rewards[-50:])/50))
#                 print("Episode: {}, Max reward:{}, Avg Reward:{}".format(episode, max(rewards[-50:]), sum(rewards[-50:])/50))
#                 print('Max Reward: {}'.format(max(rewards)))
#                 print("Average for last 10 episodes: {}".format(sum(rewards[-10:])/10))
                print(".", end="")
        print()
        return rewards


def main(num_players=2, num_episodes = 1):
    flags = {'players':num_players, 'num_episodes': num_episodes, 'agent_class':'DDQNAgent'}
    
    runner = Runner(flags)
    rewards = runner.run()
    return rewards


if __name__ == "__main__":
    num_players = 2
    num_episodes = 100000

    # logging setup
    import logging
    from datetime import datetime
    now = datetime.now()
    now_str = now.strftime('%d-%m-%Y-%H_%M_%S')
    logname = f"player-{num_players}eps-{num_episodes}hanabi"
    logging.basicConfig(filename = "./" + logname + now_str+ '.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%d-%m-%Y-%H:%M:%S',
                        level=logging.DEBUG)
    

    rewards = main(num_players=num_players, num_episodes = num_episodes)

    avg_reward = [sum(rewards[i:i+50])/50 for i in range(len(rewards)-50)]


    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.plot(list(range(len(avg_reward))), avg_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. Rewards")
    plt.savefig(logname)
    
    logging.info("Completed.")