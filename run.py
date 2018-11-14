import torch
from unityagents import UnityEnvironment
import numpy as np
from agent import PPOAgent
from collections import deque
import torch

from model import *
from config import Config

device = torch.device("cpu")

env = UnityEnvironment(file_name="Reacher.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

config = Config(num_workers=num_agents)

def run(env, brain_name, network: ActorCriticNet):
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    
    while True:
        #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions, _, _, _ = network(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
            
    return np.mean(scores)
    
def ppo(env, brain_name, network, config, weight_filename):
    weight = torch.load(weight_filename, map_location=device)
    network.load_state_dict(weight)

    score = run(env, brain_name, network)
    return [score], [score]

if __name__ == '__main__':
    weight_filename = 'ppo_checkpoint.pth'
    trained_network = ActorCriticNet(state_size=state_size, 
                                     action_size=action_size,
                                     hidden_size=512).to(device)
    
    all_scores, average_scores = ppo(env=env, brain_name=brain_name, network=trained_network, config=config, weight_filename=weight_filename)

    env.close()