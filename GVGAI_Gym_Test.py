import gym
import gym_gvgai
import numpy as np
import matplotlib.pyplot as plt


def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title('%s | Step: %d %s' %(name, step, info))
    plt.axis('off')

#print([env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')])

# state_shape = (120, 80, 4)
# action_sapce = 5

env = gym.make('gvgai-realsokoban-lvl4-v0')

env.reset()
score = 0
for i in range(100):
    print(env.action_space.n)
    show_state(env, i, 'Aliens', str(score))
    env.render()
    state, reward, isOver, debug = env.step(env.action_space.sample())
    print(state.shape[2])
    score += reward
    if(isOver):
        break