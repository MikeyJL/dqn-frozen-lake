import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))


# Parametres -----

# Total episodes
TOTAL_EPISODES = 15000
# Learning rate
LEARNING_RATE = 0.8
# Max steps per episode
MAX_STEPS = 99
# Discounting rate
GAMMA = 0.95
# Exploration probability at start
MAX_EPSILON = 1.0
# Minimum exploration probability 
MIN_EPSILON = 0.01
DECAY_RATE = 0.005  
# Decay rate
epsilon = 1.0
# List of rewards
rewards = []

# Learn -----

for episode in range(TOTAL_EPISODES):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(MAX_STEPS):
        exp_exp_tradeoff = random.uniform(0, 1)        
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(qtable[new_state, :]) - qtable[state, action])
        total_rewards += reward
        state = new_state
        
        # Breaks on death
        if done == True: 
            break
        
    # Reduce epsilon
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE*episode) 
    rewards.append(total_rewards)

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("EPISODE ", episode)

    for step in range(MAX_STEPS):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()