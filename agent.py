import gym
from utils import *
from model import ActorCritic, Memory
from algo import PPO
import gym_gvgai
import numpy as np
import matplotlib.pyplot as plt

# state_shape = (120, 80, 4)
# action_sapce = 5

############## Hyperparameters ##############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env_name = "gvgai-pacman-lvl0-v0"
# creating environment
render = False
log_interval = 20
update_timestep = 2000          # update policy every n timesteps
max_episodes = 50000
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99                    # discount factor
K_epochs = 10                   # update policy for K epochs
eps_clip = 0.2                  # clip parameter for PPO
random_seed = None
#############################################

def agent_train():
    env = gym.make(env_name)

    score = 0
    done = False
    agent = PPO(env.observation_space.shape, env.action_space.n, lr, betas, gamma, K_epochs, eps_clip, device)
    memory = Memory()

    # variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        state = preProcess(state)
        t = 0
        while done is False:
            t += 1
            timestep += 1

            #Running policy_old:
            action = agent.polciy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            state = preProcess(state)

            # Saving reward:
            memory.rewards.append(reward)

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # log
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        if i_episode % 500 == 0:
            torch.save(agent.policy.state_dict(),
                       './PPO_{}.pth'.format(env_name))

agent_train()