
# coding: utf-8

# In[9]:


import sys
import environment
from environment import MountainCar
import numpy as np

if __name__ == '__main__':
    mode_in = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes_in = sys.argv[4]
    max_iteration_in = sys.argv[5]
    epsilon_in = sys.argv[6]
    gamma_in = sys.argv[7]
    learning_rate = sys.argv[8]


class qLearning:
    def __init__(self, mode, episodes, max_iterations, epsilon, gamma, learning_rate):
        self.mode = mode 
        self.episodes = int(episodes)
        self.max_iterations = int(max_iterations)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        
        self.actions = [0,1,2]
        self.environment_mode = MountainCar(mode)
                
        self.W = np.matrix(np.zeros((self.environment_mode.action_space, self.environment_mode.state_space)))
        self.bais = 0
                
        self.all_rewards = self.iter_all_episodes()
    
    def q_learning_value(self, state, action):
        q_value = np.dot(np.matrix(state), self.W[action].T) + self.bais
        return q_value       
    
    def chooseAction(self, state):
        dist = np.random.binomial(1, self.epsilon)
        if dist == 1:
            action = np.random.randint(0, self.environment_mode.action_space)

        else:
            all_value =[]
            for action in self.actions:
                all_value.append(self.q_learning_value(state, action))
                
            action = np.argmax(all_value)

        return action
    
    def TDError(self, state, action, next_state, max_action, reward):
        TD_target = reward + self.gamma * self.q_learning_value(next_state, max_action)
        TD_error = self.q_learning_value(state, action) - TD_target
        return TD_error
    
    def update(self, state, action, next_state, max_action, reward):
        td_error = self.TDError(state, action, next_state, max_action, reward)
        self.W[action] -= (self.learning_rate * td_error * state)
        self.bais -= (self.learning_rate * td_error)
        
    
    def initializeState(self):
        if self.mode != "tile":
            env_to_list = list(self.environment_mode.reset().values())
            state = np.array(env_to_list)  
        else:
            state_key = list(self.environment_mode.reset().keys())
            state_key_array = np.array(state_key)
            state = np.matrix(np.zeros((self.environment_mode.state_space)))
            state[0,state_key] = 1
        return state
        
    
    def ForEachEpisode(self):
        curr_state = self.initializeState()
        curr_action = self.chooseAction(curr_state)
        
        cum_rewards = 0   
        itera = 0
        converge = False
        
        while (self.max_iterations > itera) and (converge == False):
            curr_action = self.chooseAction(curr_state)
            s_prime = self.environment_mode.step(curr_action)
            
            if self.mode == 'tile':
                #np.fromiter(z[0].keys(),dtype=float).astype(int)
                indx = np.fromiter(s_prime[0].keys(),dtype=float).astype(int)
                next_state = np.matrix(np.zeros((self.environment_mode.state_space)))
                next_state[0, indx] = 1
                converge = s_prime[2]
                rewards = s_prime[1]
                
            else:
                next_state = list(s_prime[0].values())
                next_state = np.array(next_state)
                converge = s_prime[2]
                rewards = s_prime[1]
        
            max_action = self.chooseAction(next_state)
            self.update(curr_state, curr_action, next_state, max_action, rewards)
            cum_rewards = cum_rewards + rewards
            curr_state = next_state
#             curr_action = max_action
            itera = itera + 1
        self.environment_mode.reset()
        return cum_rewards, converge
        
      
    def iter_all_episodes(self):
        rewards_all_episodes = []
        
        for n in range(self.episodes):
            rewards, converge = self.ForEachEpisode()
            rewards_all_episodes.append(rewards)
            
        return rewards_all_episodes
        
# mode_in = "tile"
# episodes_in = 25
# max_iteration_in = 200
# epsilon_in = 0.0
# gamma_in = 0.99
# learning_rate = 0.05

ql = qLearning(mode_in, episodes_in, max_iteration_in, epsilon_in, gamma_in, learning_rate)

weight_b = np.hstack((np.squeeze(np.asarray(ql.bais)).reshape(-1),(np.squeeze(np.asarray(ql.W.T)).reshape(-1))))
outWeight = open ('./'+weight_out,'w')
for x in range(len(weight_b)):
    text = ''
    text = str(weight_b[x])
    outWeight.writelines(text+'\n')
outWeight.close()

return_ = ql.all_rewards
outReturn = open ('./'+returns_out,'w')
for x in range(len(return_)):
    text = ''
    text = str(return_[x])
    outReturn.writelines(text+'\n')
outReturn.close()

