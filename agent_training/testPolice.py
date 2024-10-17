import torch
import numpy as np
from torch.distributions import Categorical

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TownEnv import TownEnvironment
from policy import PolicyNetwork

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
    # Sample action from the policy
    action = m.sample()

    return action.item()


def test_agent(num_episodes=100):
    env = TownEnvironment()
    input_size = 100
    output_size = 5

    policy_net = PolicyNetwork(input_size, output_size)
    policy_net.load_state_dict(torch.load('policeagent.pth', map_location='cpu', weights_only=True))
    policy_net.eval()

    total_rewards = []

    #for episode in range(num_episodes):
    state = env.reset()
    police_agent = env.add_agent('police', (0, 0))
    thief_agent = env.add_agent('thief', (9, 9))
    
    episode_reward = 0
    done = False
    max_steps = env.size * env.size * env.size

    for step in range(max_steps):
        action_index = select_action(state, policy_net)
        possible_actions = police_agent.get_action(state, env)
        action = possible_actions[action_index - 1]
        
        state, reward, done = env.step(police_agent, thief_agent, action)
        episode_reward += reward

        if done:
            break

    total_rewards.append(episode_reward)
    #print(f"Episode {episode + 1}: Reward = {episode_reward}")

    #avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {episode_reward:.2f}")

if __name__ == "__main__":
    test_agent()