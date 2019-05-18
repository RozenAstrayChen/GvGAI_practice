import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import *
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    # state_shape = (120, 80, 4)
    # action_sapce = 5
    def __init__(self, obs_shape, action_space):
        super(ActorCritic, self).__init__()

        self.nn_output = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=1),
            nn.ReLU(), # (64, 12 ,7)->5376
            Flatten(),
            nn.Linear(5376, 512),
            nn.ReLU())

        self.critic_layer = nn.Linear(512, 1)
        self.actor_layer = nn.Linear(512, action_space)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = swapAxis(state)
        state = state[np.newaxis, :]
        state = convertVariable(state)

        x = self.nn_output(state)
        action_prob = self.actor_layer(x)
        action_prob = F.softmax(action_prob, dim=-1)

        dist = Categorical(action_prob)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_layer(state)

        return action_logprobs, state_value, dist_entropy