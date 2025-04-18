import numpy as np
from utils.config import *

class ObstacleGenerator:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def generate_obstacles(self):
        """Generate fixed obstacles ensuring valid start positions"""
        obstacle_map = np.zeros(self.grid_size, dtype=np.int32)

        max_x, max_y = self.grid_size

        center_x, center_y = max_x // 2, max_y // 2
        for y in range(center_y - 3, center_y + 4):
            for x in range(center_x - 3, center_x + 4):
                # Wall around the perimeter (excluding the hollow center)
                if (x == center_x - 3 or x == center_x + 3 or  # left and right walls
                    y == center_y - 3 or y == center_y + 3):  # top and bottom walls
                    # Leave openings at the top-left and right-center
                    if not ((y == center_y - 3 and x == center_x) or 
                            (y == center_y and x == center_x + 3) or 
                            (y == center_y and x == center_x - 3) or
                            (y == center_y + 3 and x == center_x)):
                        if 0 <= y < max_y and 0 <= x < max_x:
                            obstacle_map[y, x] = 1
        
        # # L-shaped wall in bottom-left (5x5)
        # for y in range(max_y - 6, max_y-2):
        #     if 0 <= 2 < max_x:
        #         obstacle_map[y, 2] = 1
        # for x in range(2, 7):
        #     if 0 <= max_y - 3 < max_y:
        #         obstacle_map[max_y - 2, x] = 1

        # Ensure start positions are clear
        obstacle_map[0:2, 0:2] = 0  # Police station area
        obstacle_map[0:2, max_x - 2:max_x] = 0  # Bank area

        return obstacle_map
    def is_valid_position(self, pos, obstacle_map):
            """Check if position is within bounds and not obstacle"""
            x, y = pos
            max_x, max_y = self.grid_size
            return (
                0 <= x < max_x and
                0 <= y < max_y and
                obstacle_map[y, x] == 0
            )
# import numpy as np
# from utils.config import *

# class ObstacleGenerator:
#     def __init__(self, grid_size):
#         self.grid_size = grid_size  # This should be a tuple (width, height)

#     def generate_obstacles(self):
#         """Generate random obstacles ensuring valid start positions"""
#         obstacle_map = np.zeros(self.grid_size, dtype=np.int32)

#         max_x, max_y = self.grid_size

#         # Add 8-10 obstacles of different types
#         for _ in range(np.random.randint(8, 11)):
#             obs_type = np.random.randint(0, 3)
#             x = np.random.randint(1, max_x - 3)
#             y = np.random.randint(1, max_y - 3)
#             obs_size = np.random.randint(1, 6)  # Size of the obstacle (1-5 blocks)

#             if obs_type == 0:  # L-shape
#                 obstacle_map[y:y+obs_size-1, x] = 1
#                 obstacle_map[y, x:x+obs_size-1] = 1
#             elif obs_type == 1:  # Horizontal
#                 obstacle_map[y, x:x+obs_size] = 1
#             else:  # Vertical
#                 obstacle_map[y:y+obs_size, x] = 1

#         # Ensure start positions are clear
#         obstacle_map[0:2, 0:2] = 0  # Police station area
#         obstacle_map[0:2, max_x-2:max_x] = 0  # Bank area

#         return obstacle_map

#     def is_valid_position(self, pos, obstacle_map):
#         """Check if position is within bounds and not obstacle"""
#         x, y = pos
#         max_x, max_y = self.grid_size
#         return (
#             0 <= x < max_x and
#             0 <= y < max_y and
#             obstacle_map[y, x] == 0
#         )