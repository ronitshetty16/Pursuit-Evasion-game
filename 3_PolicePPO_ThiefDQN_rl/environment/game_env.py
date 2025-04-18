import numpy as np
import time
from gym import spaces
from utils.config import *
from utils.visualization import GameVisualizer
from environment.obstacles import ObstacleGenerator

class PoliceThiefEnv:
    def __init__(self, visualize=False):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.episode_duration = EPISODE_DURATION
        self.visualize = visualize
        self.time_speed = TIME_SPEED_MULTIPLIER
        
        # Initialize components
        self.obstacle_generator = ObstacleGenerator(self.grid_size)
        self.visualizer = GameVisualizer(self.grid_size, self.cell_size) if visualize else None
        
        # Action space: 8 directions
        self.action_space = spaces.Discrete(8)
        
        # Observation space (positions only)
        self.observation_space = spaces.Dict({
            "police_obs": spaces.Dict({
                "my_position": spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,), dtype=np.int32),
                "target_position": spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,), dtype=np.int32)
            }),
            "thief_obs": spaces.Dict({
                "my_position": spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,), dtype=np.int32),
                "target_position": spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,), dtype=np.int32)
            })
        })
        
        # Initialize game state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.police_pos = np.array([1, 1])  # y,x (row,col)
        self.thief_pos = np.array([1, self.grid_size[0]-2])  # y,x
        self.obstacle_map = self.obstacle_generator.generate_obstacles()
        self.start_time = time.time()
        self.elapsed_time = 0
        self.episode_rewards = [0, 0]
        return self._get_observation()
    
    def _get_observation(self):
        """Return current observation"""
        return {
            "police_obs": {
                "my_position": self.police_pos,
                "target_position": self.thief_pos
            },
            "thief_obs": {
                "my_position": self.thief_pos,
                "target_position": self.police_pos
            }
        }
    
    def _is_valid_move(self, new_pos):
        """Check if move is within bounds and not obstacle"""
        y, x = new_pos  # Note: y is row, x is column
        return (0 <= x < self.grid_size[0] and  # Check x within width
                0 <= y < self.grid_size[1] and  # Check y within height
                self.obstacle_map[y,x] == 0)
    
    def step(self, police_action=None, thief_action=None, episode=0):
        """Execute one time step"""
        real_elapsed = time.time() - self.start_time
        self.elapsed_time = real_elapsed * self.time_speed
        
        # Action mappings (8 directions)
        moves = [(-1,-1),(-1,0),(-1,1),
                (0,-1),        (0,1),
                (1,-1), (1,0), (1,1)]
        
        # Move agents with collision checking
        if police_action is not None:
            new_pos = self.police_pos + moves[police_action]
            if self._is_valid_move(new_pos):
                self.police_pos = new_pos
                
        if thief_action is not None:
            new_pos = self.thief_pos + moves[thief_action]
            if self._is_valid_move(new_pos):
                self.thief_pos = new_pos
        
        # Calculate rewards and done status
        done = False
        police_reward = POLICE_TIME_PENALTY
        thief_reward = THIEF_ESCAPE_REWARD
        
        if np.array_equal(self.police_pos, self.thief_pos):
            police_reward = POLICE_CATCH_REWARD
            thief_reward = THIEF_CAUGHT_PENALTY
            done = True
        elif self.elapsed_time >= self.episode_duration:
            done = True
            if not np.array_equal(self.police_pos, self.thief_pos):
                police_reward += POLICE_FAIL_PENALTY
                thief_reward += THIEF_SURVIVAL_REWARD
        
        # Update and render
        self.episode_rewards[0] += police_reward
        self.episode_rewards[1] += thief_reward
        
        if self.visualize:
            self.render(episode=episode)
            
        return self._get_observation(), (police_reward, thief_reward), done, {}
    
    def render(self, episode=0, speed=1.0):
        """Render the current state"""
        if self.visualizer:
            return self.visualizer.render(
                police_pos=self.police_pos,
                thief_pos=self.thief_pos,
                obstacle_map=self.obstacle_map,
                elapsed_time=self.elapsed_time,
                episode=episode,
                police_reward=self.episode_rewards[0],
                thief_reward=self.episode_rewards[1],
                speed=speed
            )
        return True
    
    def close(self):
        """Clean up resources"""
        if self.visualizer:
            self.visualizer.close()