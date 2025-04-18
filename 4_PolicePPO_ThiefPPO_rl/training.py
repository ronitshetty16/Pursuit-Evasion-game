import numpy as np
import torch
from environment.game_env import PoliceThiefEnv
from agents.police_ppo import PolicePPOAgent  
from agents.thief_ppo import ThiefPPOAgent
from utils.config import *
import time
import matplotlib.pyplot as plt
import os

def train_agents(visualize_training=True, render_speed=2.0):
    # Initialize environment with visualization
    env = PoliceThiefEnv(visualize=visualize_training)
    police_agent = PolicePPOAgent(env)
    thief_agent = ThiefPPOAgent(env)
    
    # Training parameters
    batch_size = 64
    episodes = 3000

    # #Testing batch
    # batch_size = 8
    # episodes = 100
    
    # For tracking rewards and losses
    police_rewards = []
    thief_rewards = []
    police_losses = []
    thief_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_police_reward = 0
        episode_thief_reward = 0
        done = False
        
        while not done:
            # Agents choose actions
            police_action = police_agent.act(state)
            thief_action = thief_agent.act(state)
            
            # Take step in environment
            next_state, (police_reward, thief_reward), done, _ = env.step(
                police_action, thief_action, episode=episode)
            
            # Store experience for PPO updates
            state_tensor = torch.FloatTensor(police_agent._preprocess_state(state))
            with torch.no_grad():
                police_value = police_agent.value(state_tensor).item()
                thief_value = thief_agent.value(state_tensor).item()
            
            police_agent.remember(police_reward, police_value, done)
            thief_agent.remember(thief_reward, thief_value, done)
            
            # Train agents
            police_loss = police_agent.train()
            thief_loss = thief_agent.train()
            
            # Track losses
            if police_loss is not None:
                police_losses.append(police_loss)
            if thief_loss is not None:
                thief_losses.append(thief_loss)
            
            # Update tracking variables
            episode_police_reward += police_reward
            episode_thief_reward += thief_reward
            state = next_state
            
            # Render at specified speed if visualization enabled
            if visualize_training:
                env.render(speed=render_speed)
                
        # Store episode rewards
        police_rewards.append(episode_police_reward)
        thief_rewards.append(episode_thief_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_police_loss = np.mean(police_losses[-10:]) if police_losses else 0
            avg_thief_loss = np.mean(thief_losses[-10:]) if thief_losses else 0
            print(f"Episode {episode}: "
                  f"Police Reward: {episode_police_reward:.2f} (Loss: {avg_police_loss:.4f}), "
                  f"Thief Reward: {episode_thief_reward:.2f} (Loss: {avg_thief_loss:.4f})")
    
        def moving_average(data, window_size=64):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Plot results
        plt.figure(figsize=(12, 6))

        # --- Rewards Plot ---
        plt.subplot(1, 2, 1)
        plt.plot(police_rewards, color='lightblue', label='Police (raw)')
        plt.plot(thief_rewards, color='lightcoral', label='Thief (raw)')
        if len(police_rewards) >= 64:
            plt.plot(moving_average(police_rewards), color='blue', label='Police (avg)')
        if len(thief_rewards) >= 64:
            plt.plot(moving_average(thief_rewards), color='red', label='Thief (avg)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()

        # --- Losses Plot ---
        plt.subplot(1, 2, 2)
        plt.plot(police_losses, color='lightblue', label='Police Loss (raw)')
        plt.plot(thief_losses, color='lightcoral', label='Thief Loss (raw)')
        if len(police_losses) >= 64:
            plt.plot(moving_average(police_losses), color='blue', label='Police Loss (avg)')
        if len(thief_losses) >= 64:
            plt.plot(moving_average(thief_losses), color='red', label='Thief Loss (avg)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('plots/training_results.png')
        plt.close()

    # Save models
    torch.save({
        'policy': police_agent.policy.state_dict(),
        'value': police_agent.value.state_dict()
    }, 'models/police_ppo.pth')
    torch.save({
        'policy': thief_agent.policy.state_dict(),
        'value': thief_agent.value.state_dict()
    }, 'models/thief_ppo.pth')
    
    return police_agent, thief_agent

if __name__ == "__main__":
    police_agent, thief_agent = train_agents(visualize_training=True, render_speed=200.0)

