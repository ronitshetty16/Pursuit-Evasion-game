import pygame
import numpy as np
from utils.config import *

class GameVisualizer:
    def __init__(self, grid_size, cell_size):
        pygame.init()
        self.grid_width, self.grid_height = grid_size  # (20, 15)
        self.cell_size = cell_size
        self.screen_width = self.grid_width * cell_size
        self.screen_height = self.grid_height * cell_size + 60  # +60 for info panel
        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Police vs Thief - City Chase")
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.clock = pygame.time.Clock()
        
        # Colors (from config)
        self.colors = {
            'background': BACKGROUND_COLOR,
            'road': ROAD_COLOR,
            'building': OBSTACLE_COLOR,
            'police': POLICE_COLOR,
            'thief': THIEF_COLOR,
            'police_station': POLICE_STATION_COLOR,
            'bank': BANK_COLOR
        }
        
        # Create sprites
        self._create_sprites()

    def _create_sprites(self):
        """Create all visual elements"""
        # Police station (3x3 cells)
        self.police_station = pygame.Surface((self.cell_size*3, self.cell_size*3))
        self.police_station.fill(self.colors['police_station'])
        text = self.font.render("RLPD", True, (255,255,255))
        text_rect = text.get_rect(center=(self.cell_size*1.5, self.cell_size*1.5))
        self.police_station.blit(text, text_rect)
        
        # Bank (3x3 cells)
        self.bank = pygame.Surface((self.cell_size*3, self.cell_size*3))
        self.bank.fill(self.colors['bank'])
        text = self.font.render("$$$", True, (0,0,0))
        text_rect = text.get_rect(center=(self.cell_size*1.5, self.cell_size*1.5))
        self.bank.blit(text, text_rect)
        
        # Agent sprites
        self.police_sprite = self._create_agent_sprite(self.colors['police'], "P")
        self.thief_sprite = self._create_agent_sprite(self.colors['thief'], "T")

    def _create_agent_sprite(self, color, label):
        """Create human-like agent sprite"""
        surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        
        # Draw head (top circle)
        pygame.draw.circle(surf, color, 
                          (self.cell_size//2, self.cell_size//3), 
                          self.cell_size//4)
        
        # Draw body (bottom rectangle)
        pygame.draw.rect(surf, color,
                        (self.cell_size//3, self.cell_size//2,
                         self.cell_size//3, self.cell_size//2))
        
        # Add label
        text = self.small_font.render(label, True, (255,255,255))
        text_rect = text.get_rect(center=(self.cell_size//2, self.cell_size//2))
        surf.blit(text, text_rect)
        
        return surf

    def render(self, police_pos, thief_pos, obstacle_map, elapsed_time, episode, police_reward, thief_reward, speed=1.0):
        """Render the current game state"""
        # Handle events (close window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # # Draw grid background (roads)
        # for y in range(self.grid_height):
        #     for x in range(self.grid_width):
        #         rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
        #                           self.cell_size, self.cell_size)
        #         pygame.draw.rect(self.screen, self.colors['road'], rect, 1)
        
        # Draw obstacles (buildings)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if obstacle_map[y,x] == 1:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, self.colors['building'], rect)
        
        # Draw landmarks
        self.screen.blit(self.police_station, (0, 0))  # Top-left
        self.screen.blit(self.bank, ((self.grid_width-3)*self.cell_size, 0))  # Top-right
        
        # Draw agents - convert (row,col) to (x,y) for rendering
        police_x, police_y = police_pos[1], police_pos[0]  # (col,row)
        thief_x, thief_y = thief_pos[1], thief_pos[0]       # (col,row)
        self.screen.blit(self.police_sprite, (police_x*self.cell_size, police_y*self.cell_size))
        self.screen.blit(self.thief_sprite, (thief_x*self.cell_size, thief_y*self.cell_size))
        
        # Draw info panel
        self._draw_info_panel(elapsed_time, episode, police_reward, thief_reward)
        
        pygame.display.flip()
        self.clock.tick(FPS * speed)
        return True

    def _draw_info_panel(self, elapsed_time, episode, police_reward, thief_reward):
        """Draw information panel at bottom"""
        panel_y = self.grid_height * self.cell_size
        pygame.draw.rect(self.screen, (220,220,220), (0, panel_y, self.screen_width, 60))
        
        # Create info texts
        time_remaining = max(0, EPISODE_DURATION - elapsed_time)
        texts = [
            f"Episode: {episode}",
            f"Time: {time_remaining:.1f}s",
            f"Police: {police_reward:.1f}",
            f"Thief: {thief_reward:.1f}"
        ]
        
        # Draw texts
        for i, text in enumerate(texts):
            color = (0,0,0) if i < 2 else (self.colors['police'] if i == 2 else self.colors['thief'])
            text_surf = self.font.render(text, True, color)
            x_pos = 10 + (i%2)*300
            y_pos = panel_y + 5 + (i//2)*25
            self.screen.blit(text_surf, (x_pos, y_pos))

    def close(self):
        pygame.quit()