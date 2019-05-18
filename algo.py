import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic


class PPO():
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.polciy_old = ActorCritic(state_dim, action_dim).to(self.device)

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards
        rewards = []
        discount_reward = 0
        for reward in reversed(memory.rewards):
            discount_reward = reward + (self.gamma * discount_reward)
            rewards.insert(0, discount_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list in tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old acions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            loss = - torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())