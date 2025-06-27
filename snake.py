from collections import deque  
import numpy as np
import matplotlib.pyplot as plt
import random


class Snake:
    def __init__(self, grid_size):  
        self.observation_space = (grid_size, grid_size)
        self.action_space = [0, 1, 2, 3]  # up, down, left, right
        self.action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        self.grid_size = grid_size
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.reset()

    def get_state(self):
        return (
            self.snake,
            self.mice,
            self.done,
            self.score,
        )

    def reset(self):     
        self.done = False
        self.score = 0
        self.snake = deque([(self.grid_size//2, self.grid_size//2)])
        self.spawn_mice()
        return self.get_state(), 0, False, {}

    def step(self, action):  
        dx, dy = self.action_map[action]

        # Check if we take action and snake dies?
        # adds the action to the current position
        new_head_x = self.snake[0][0] + dx
        # adds the action to the current position
        new_head_y = self.snake[0][1] + dy

        if (
            new_head_x < 0 or new_head_y < 0 or 
            new_head_x >= self.grid_size or new_head_y >= self.grid_size
        ):
            self.done = True
            return self.get_state(), -1, self.done, {}  # state, reward, done, info

        # check if snake ate itself
        if (new_head_x, new_head_y) in self.snake:
            self.done = True
            return self.get_state(), -1, self.done, {}  # state, reward, done, info

        # If snake does not die and its a valid move
        # move snake
        self.snake.appendleft((new_head_x, new_head_y))

        # check if snake ate the mice
        if (new_head_x, new_head_y) == self.mice:
            self.score += 1
            self.spawn_mice()  # spawns new mice
            return self.get_state(), 1, self.done, {}
        else:
            self.snake.pop()  # remove tail - simulates movement
            # to encourage faster food acquisition we will reinforce a small penalty for every step
            return self.get_state(), -0.01, self.done, {}

    def env_to_matrix(self):       
        # Create a gridxgrid matrix of all zeros
        grid = np.zeros((self.grid_size, self.grid_size))
        # puts 1 wherever the snake is
        for x, y in self.snake:
            grid[x, y] = 1

        # puts 2 where the mice is
        grid[self.mice[0], self.mice[1]] = 2
        self.grid = grid
    
    def render(self):   
        self.env_to_matrix()
        return self.grid

    def spawn_mice(self):
        # should check for snake position
        all_positions = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        )
        occupied_by_snake = set(self.snake)
        free_positions = all_positions - occupied_by_snake
        if not free_positions:
            self.done = True
            self.mice = None
        else:
            self.mice = random.choice(list(free_positions))
        return self.mice
