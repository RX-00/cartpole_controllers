import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from collections import deque



# ---------- Setting up the environment ------------
'''
    Different Environments of Interest:
    Bipedal Walking
    Hopper
    Pendulum
    CartPole
'''
env_id = "CartPole-v1" # should work plug-n-play for other environments too

env = gym.make(env_id)

state_dim = env.observation_space.shape[0] # state space size
action_dim = env.action_space.n             # action space size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# -------- Actor & Critic NN Combined Object -------
class ActorCritic(nn.Module):
    '''
        NOTE: has_cont_as refers to whether or not the environment has a continuous action space
              init_as_std is the initial action-space standard deviation
    '''
    def __init__(self, state_dim, action_dim, has_cont_as, init_as_std):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_cont_as = has_cont_as
        self.as_std = init_as_std
        # set action variance matrix
        if self.has_cont_as:
            self.action_var = torch.full((action_dim), self.as_std * self.as_std).to(device)

        # NOTE: we use linear layers since they're good for stabilization of dynamic systems
        #       also use tanh instead of relu due to dying neuron problem

        # Actor NN -> continuous or discrete action-space
        #           model performs the task of learning what action to take under state observation of env
        if self.has_cont_as:
            self.actor = nn.Sequential(nn.Linear(self.state_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, self.action_dim))
            
        else:
            self.actor = nn.Sequential(nn.Linear(self.state_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, self.action_dim),
                                       nn.Softmax(dim=1)) # make sure the rows sum to 1 for discretized action-space
            
        # Critic NN -> doesn't have to worry about output discretization (or continuinity)
        #           evaluate if the action taken leads to a better state. Outputs a rating (Q-value) of action taken in prev. state
        self.critic = nn.Sequential(nn.Linear(self.state_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, self.action_dim))
        
    def forward(self):
        raise NotImplementedError
        
    # helper function to change the continuous action-space standard deviation
    def set_as_std(self, new_as_std):
        if self.has_cont_as:
            self.as_std = new_as_std
            self.action_var = torch.full((self.action_dim), self.as_std * self.as_std).to(device)
        else:
            print("WARN: You're calling ActorCritic::set_as_std() on discrete action-space system")

    def act(self, state):
        if self.has_cont_as:
            action_mean = self.actor(state)
            covar_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            distr = MultivariateNormal(action_mean, covar_mat)
        else:
            action_probs = self.actor(state)
            distr = Categorical(action_probs)

        action = distr.sample()
        action_logprob = distr.log_prob(action)
        state_val = self.critic(state)

        # return new tensors detached from the current graph with no grad
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    # function used to evaluate values of prev. states & actions
    def evaluate(self, state, action):
        if self.has_cont_as:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            covar_mat = torch.diag_embed(action_var).to(device)
            distr = MultivariateNormal(action_mean, covar_mat)

            # in the case for continuous action-space with single input
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            distr = Categorical(action_probs)

        action_logprobs = distr.log_prob(action)
        state_values = self.critic(state)
        distr_entropy = distr.entropy()

        return action_logprobs, state_values, distr_entropy



# ----------- Rollout Buffer of Episodes -----------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    # clear the buffer
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]



# --------------- PPO Implementation ---------------
class PPO:
    def __init__(self, ):
        