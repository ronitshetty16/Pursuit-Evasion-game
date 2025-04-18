import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.game_env import PoliceThiefEnv
from agents.police_dqn import PoliceDQNAgent
from agents.thief_ppo import ThiefPPOAgent
from utils.config import *

def load_trained_agents(env):
    # Initialize agents
    police_agent = PoliceDQNAgent(env)
    thief_agent = ThiefPPOAgent(env)

    # Load police DQN model
    police_agent.model.load_state_dict(torch.load("police_dqn.pth"))
    police_agent.model.eval()

    # Load thief PPO policy and value networks
    checkpoint = torch.load("thief_ppo.pth")
    thief_agent.policy.load_state_dict(checkpoint["policy"])
    thief_agent.value.load_state_dict(checkpoint["value"])
    thief_agent.policy.eval()
    thief_agent.value.eval()

    return police_agent, thief_agent

def evaluate_agents(episodes=50, visualize=False, render_speed=1.0):
    env = PoliceThiefEnv(visualize=visualize)
    police_agent, thief_agent = load_trained_agents(env)

    police_rewards = []
    thief_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_police_reward = 0
        episode_thief_reward = 0

        while not done:
            # Get actions (no exploration here)
            police_action = police_agent.act(state, training=False)
            thief_action = thief_agent.act(state)

            # Step through environment
            next_state, (police_reward, thief_reward), done, _ = env.step(
                police_action, thief_action, episode=episode)

            episode_police_reward += police_reward
            episode_thief_reward += thief_reward
            state = next_state

            if visualize:
                env.render(speed=render_speed)

        police_rewards.append(episode_police_reward)
        thief_rewards.append(episode_thief_reward)

        print(f"Episode {episode + 1}: Police Reward = {episode_police_reward:.2f}, Thief Reward = {episode_thief_reward:.2f}")

    env.close()
    return police_rewards, thief_rewards

def plot_rewards(police_rewards, thief_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(police_rewards, label='Police', color='blue')
    plt.plot(thief_rewards, label='Thief', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_rewards.png")
    plt.show()

if __name__ == "__main__":
    police_rewards, thief_rewards = evaluate_agents(
        episodes=50, visualize=True, render_speed=50.0)
    plot_rewards(police_rewards, thief_rewards)
