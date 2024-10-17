import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque
from TownEnv import TownEnvironment
from policy import PolicyNetwork
import torch
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_model(model, fig_size=(12, 8)):
    # Visualization logic here...
    pass

def select_action(state, policy_net, epsilon=0.1):
    state_tensor = torch.FloatTensor(state).flatten().detach()
    
    # Ensure the input requires gradients
    state_tensor.requires_grad_(True)
    
    probs = policy_net(state_tensor)

    if torch.isnan(probs).any() or (probs < 0).any() or probs.sum() == 0:
        print("Invalid probabilities detected:", probs)
        # Reset to uniform distribution
        probs = torch.ones_like(probs) / len(probs)
    else:
        # Normalize probabilities
        probs = probs / probs.sum()

    # Create Categorical distribution
    m = Categorical(probs)

    if random.random() < epsilon:
        # Epsilon-greedy exploration
        action = torch.randint(0, len(probs), (1,))
        log_prob = m.log_prob(action)
    else:
        # Sample action from the policy
        action = m.sample()
        log_prob = m.log_prob(action)

    policy_net.saved_log_probs.append(log_prob)

    return action.item()

def finish_episode(eps, optimizer, policy_net):
    gamma = 0.8
    R = 0
    policy_loss = []
    returns = deque()
    
    for r in policy_net.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    
    returns = torch.tensor(returns)
    if len(returns)>1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for log_prob, R in zip(policy_net.saved_log_probs, returns):
        policy_loss.append(-log_prob * R.unsqueeze(0))
    
    optimizer.zero_grad()
    if policy_loss:
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
    optimizer.step()
    
    del policy_net.rewards[:]
    del policy_net.saved_log_probs[:]
    return 0

def train(env, agent,targetagent, policy_net, optimizer, num_episodes=100000):
    running_reward = 100
    max_steps_per_episode = env.size * env.size * env.size
    state = env.get_state()
    pos = (0,0)
    for episode in range(num_episodes):
        if episode%100 == 0:
            state = env.reset()
        elif episode%10==0:
            env.resetagent(agent)
            pos = agent.pos
        else:
            env.resetagentpos(agent,pos)
        ep_reward = 0
        policy_net.rewards = []
        policy_net.saved_log_probs = []
        
        #env.render()
        done = False
        for t in range(1, max_steps_per_episode):
            action_index = select_action(state, policy_net=policy_net)
            possible_actions= agent.get_action(state,env)
            action = possible_actions[action_index-1]
            state, reward, done = env.step(agent,targetagent, action)
            
            policy_net.rewards.append(reward)
            ep_reward += reward
            if episode%1000 == 999:
                env.render()
            
            if done:
                break
            
        running_reward = ep_reward
        finish_episode(episode, optimizer, policy_net)

        if episode%100==99:
            print(f"Episode {episode}, Running Reward: {running_reward:.2f}")
        
        if episode % 1000 == 999:
            torch.save(policy_net.state_dict(), 'policeagent.pth')
            print("Saved model", episode)
            #env.resetagent(targetagent)
        

def train_agent(policy_net=None):
    model_path = 'policeagent.pth'
    input_size = 100
    output_size = 5  # Number of possible actions

    # Initialize or load the policy network
    if not policy_net:
        policy_net = PolicyNetwork(input_size, output_size)
    
    # Check if the model file exists and load it safely
    if os.path.exists(model_path):
        print("Loading existing model...")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        policy_net.load_state_dict(state_dict)
    else:
        print("No existing model found. Starting fresh.")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)

    env = TownEnvironment()
    policeagent = env.add_agent('police',(0,0))
    thiefagent = env.add_agent('thief',(9,9))
    
    train(env, policeagent, thiefagent, policy_net, optimizer, num_episodes=100000)

    torch.save(policy_net.state_dict(), 'policeagent.pth')

if __name__ == "__main__":
    train_agent()