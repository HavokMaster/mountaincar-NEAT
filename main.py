"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle
import gymnasium as gym
import neat
import numpy as np

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make('MountainCar-v0', render_mode = "human")
done = False
state = env.reset()[0]

while not done:
    action = np.argmax(net.activate(state))
    state, reward, done, _, _ = env.step(action)