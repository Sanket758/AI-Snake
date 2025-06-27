from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtracting the max value for numerical stability
    return e_x / e_x.sum()


def animate_episode(episode_grids, interval=200):
    fig = plt.figure(figsize=(5, 5))

    def update(frame):
        plt.clf()
        plt.imshow(episode_grids[frame])
        plt.title(f"Frame {frame}")
        plt.axis("off")
        return (plt.gca(),)

    anim = FuncAnimation(fig, update, frames=len(episode_grids), interval=interval)

    # Save as gif
    anim.save("snake_game.gif", writer="pillow")

    # Or save as mp4 (requires ffmpeg)
    # anim.save('snake_game.mp4', writer='ffmpeg')

    return anim


# Action mapping for display purposes
action_to_str = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

# Wall direction mapping
wall_direction_to_str = {
    0: "top",
    1: "bottom",
    2: "left",
    3: "right",
}


class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

        self.action_space = [0, 1, 2, 3]  # up, down, left, right
        self.action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        self.feature_names = [
            "dx_to_food",
            "dy_to_food",
            "dir_x_normalized",
            "dir_y_normalized",
            "up_indicator",
            "down_indicator",
            "left_indicator",
            "right_indicator",
            "distance_from_mice",
            "distance_from_wall",
            "direction_from_wall",
            "snake_length",
        ]

    def _get_snake_distance_from_mice(self):
        dy = (self.mice[0] - self.snake[0][0]) / self.grid_size  # dx_to_food
        dx = (self.mice[1] - self.snake[0][1]) / self.grid_size  # dt_to_food
        return dx, dy

    def build_observation_space(self):
        """
        dx, dy = distance from snake head to mice
        distance_from_mice = manhatten_distance from mice to snake head
        distance_from_wall = manhatten_distance from wall to snake head
        direction_from_wall = direction from wall to snake head
        snake_length = length of the snake vector
        """
        distance_from_mice = manhattan(self.snake[0], self.mice)
        # Calculate direct distances to each wall
        snake_y, snake_x = self.snake[0]

        # Distance to each wall (top, bottom, left, right)
        distances_from_wall = [
            snake_y,  # Distance to top wall (0)
            self.grid_size - 1 - snake_y,  # Distance to bottom wall (1)
            snake_x,  # Distance to left wall (2)
            self.grid_size - 1 - snake_x,  # Distance to right wall (3)
        ]

        # Print for debugging
        # print(f"Snake head position: {self.snake[0]}")
        # print(f"distances_from_wall: {distances_from_wall}")

        # Find the minimum distance and corresponding direction
        distance_from_wall = min(distances_from_wall)
        direction_from_wall = np.argmin(distances_from_wall)

        dx, dy = self._get_snake_distance_from_mice()
        magnitude = max(np.sqrt(dx**2 + dy**2), 1e-5)
        dir_x, dir_y = dx / magnitude, dy / magnitude

        # Create directional indicators (1 if food is in that direction, 0 otherwise)
        # These help the agent learn which direction to move more explicitly
        snake_y, snake_x = self.snake[0]
        mice_y, mice_x = self.mice
        # print(snake_x, snake_y, mice_x, mice_y)
        up_indicator = 1.0 if mice_y < snake_y else 0.0
        down_indicator = 1.0 if mice_y > snake_y else 0.0
        left_indicator = 1.0 if mice_x < snake_x else 0.0
        right_indicator = 1.0 if mice_x > snake_x else 0.0

        snake_length = len(self.snake)
        self.observation_space = (
            dx,
            dy,
            dir_x,
            dir_y,
            up_indicator,
            down_indicator,
            left_indicator,
            right_indicator,
            distance_from_mice,
            distance_from_wall,
            direction_from_wall,
            snake_length,
        )
        # print('self.observation_space: ', self.observation_space)
        return np.array(self.observation_space)

    def get_state(self):
        self.observation_space = self.build_observation_space()
        return self.observation_space

    def reset(self):
        self.done = False
        self.score = 0
        self.snake = deque(
            [
                (
                    np.random.randint(0, self.grid_size - 1),
                    np.random.randint(0, self.grid_size - 1),
                )
            ]
        )
        self.spawn_mice()
        self.observation_space = self.build_observation_space()
        return self.get_state(), 0, False, {}

    def step(self, action):
        dx, dy = self.action_map[action]

        # Check if we take action and snake dies?
        # adds the action to the current position
        new_head_x = self.snake[0][0] + dx
        # adds the action to the current position
        new_head_y = self.snake[0][1] + dy

        if (
            new_head_x < 0
            or new_head_y < 0
            or new_head_x >= self.grid_size
            or new_head_y >= self.grid_size
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
            # if snake moves closer to mice, give slightly positive reward else slightly negative reward
            prev_distance = manhattan(self.snake[0], self.mice)
            new_distance = manhattan((new_head_x, new_head_y), self.mice)
            if new_distance < prev_distance:
                reward = 0.02
            else:
                reward = -0.01
            return self.get_state(), reward, self.done, {}

    def render(self):
        # Create a gridxgrid matrix of all zeros
        grid = np.zeros((self.grid_size, self.grid_size))
        # puts 1 wherever the snake is
        for x, y in self.snake:
            grid[x, y] = 1

        # puts 2 where the mice is
        grid[self.mice[0], self.mice[1]] = 2
        return grid

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

    def __repr__(self):
        """Return a string representation of the Snake environment."""
        head_pos = self.snake[0] if self.snake else "None"
        return (
            f"Snake(grid_size={self.grid_size}, "
            f"score={self.score}, "
            f"snake_length={len(self.snake)}, "
            f"head_pos={head_pos}, "
            f"mice_pos={self.mice}, "
            f"done={self.done})"
        )

    def display_state(self, state=None):
        """
        Display the state in a human-readable format with feature names.

        Args:
            state: The state to display. If None, uses the current state.

        Returns:
            A formatted string representation of the state.
        """
        if state is None:
            state = self.observation_space

        # Ensure state is a numpy array
        state = np.array(state)

        # Create a formatted string with feature names and values
        lines = []
        lines.append("=" * 50)
        lines.append("STATE REPRESENTATION".center(50))
        lines.append("=" * 50)

        # Calculate the maximum length of feature names for alignment
        max_name_length = max(len(name) for name in self.feature_names)

        # Group related features
        groups = {
            "Position Relative to Food": [
                "dx_to_food",
                "dy_to_food",
                "dir_x_normalized",
                "dir_y_normalized",
            ],
            "Direction Indicators": [
                "up_indicator",
                "down_indicator",
                "left_indicator",
                "right_indicator",
            ],
            "Distance Metrics": [
                "distance_from_mice",
                "distance_from_wall",
                "direction_from_wall",
            ],
            "Snake Properties": ["snake_length"],
        }

        # Display features by group
        for group_name, feature_list in groups.items():
            lines.append(f"\n{group_name}:")
            lines.append("-" * 50)

            for feature in feature_list:
                idx = self.feature_names.index(feature)
                if idx < len(state):
                    value = state[idx]
                    # Format the value based on its type
                    if isinstance(value, (int, np.integer)):
                        value_str = f"{value}"
                    else:
                        value_str = f"{value:.4f}"

                    # Add color indicators for binary features (0 or 1)
                    if feature.startswith(
                        ("up_", "down_", "left_", "right_", "danger_")
                    ):
                        if value > 0.5:  # Activated
                            indicator = "✓"
                        else:  # Not activated
                            indicator = "✗"
                        lines.append(
                            f"  {feature.ljust(max_name_length)}: {value_str} {indicator}"
                        )
                    else:
                        if feature == "direction_from_wall":
                            wall_direction = wall_direction_to_str.get(
                                int(value), "unknown"
                            )
                            lines.append(
                                f"  {feature.ljust(max_name_length)}: {value_str} {wall_direction}"
                            )
                        elif feature.startswith("direction_"):
                            lines.append(
                                f"  {feature.ljust(max_name_length)}: {value_str} {action_to_str[value]}"
                            )
                        else:
                            lines.append(
                                f"  {feature.ljust(max_name_length)}: {value_str}"
                            )

        lines.append("\n" + "=" * 50)

        # Join all lines and return
        return "\n".join(lines)


def test_wall_distance_calculation():
    """Test function to verify wall distance calculations."""
    env = Snake(grid_size=9)

    # Test specific positions
    test_positions = [
        (0, 6),  # Top wall
        (8, 3),  # Bottom wall
        (4, 0),  # Left wall
        (2, 8),  # Right wall
        (0, 0),  # Top-left corner
        (8, 8),  # Bottom-right corner
        (4, 4),  # Center
    ]

    for pos in test_positions:
        # Set snake head position manually
        env.snake = deque([pos])

        # Calculate distances
        snake_y, snake_x = pos
        distances = [
            snake_y,  # Distance to top wall
            env.grid_size - 1 - snake_y,  # Distance to bottom wall
            snake_x,  # Distance to left wall
            env.grid_size - 1 - snake_x,  # Distance to right wall
        ]

        # Get state
        state = env.get_state()

        print(f"\nSnake head at {pos}:")
        print(f"Distances to walls [top, bottom, left, right]: {distances}")
        print(f"Minimum distance: {min(distances)}")
        print(f"Closest wall: {wall_direction_to_str[np.argmin(distances)]}")
        print(env.display_state(state))

        # Render grid
        plt.figure(figsize=(4, 4))
        plt.imshow(env.render())
        plt.title(f"Snake at {pos}")
        plt.show()


if __name__ == "__main__":
    # Uncomment to test wall distance calculation
    # test_wall_distance_calculation()

    # Test the environment
    env = Snake(grid_size=9)
    state, _, _, _ = env.reset()

    # Display environment information
    print(env)  # Uses the __repr__ method

    # Display the state in a human-readable format
    print(env.display_state(state))

    # Show the grid
    print("Grid:")
    plt.figure(figsize=(6, 6))
    plt.imshow(env.render())
    plt.title("Initial State")
    plt.show()

    # Take a random action
    action_idx = random.choice(env.action_space)
    action_name = ["up", "down", "left", "right"][action_idx]
    next_state, reward, done, _ = env.step(action_idx)

    print(f"\nAction taken: {action_name} (index: {action_idx})")
    print(f"Reward: {reward}, Done: {done}")

    # Display the new state
    print(env.display_state(next_state))

    # Show the updated grid
    plt.figure(figsize=(6, 6))
    plt.imshow(env.render())
    plt.title(f"After {action_name} action")
    plt.show()

    # Take another random action
    action_idx = random.choice(env.action_space)
    action_name = ["up", "down", "left", "right"][action_idx]
    next_state, reward, done, _ = env.step(action_idx)

    print(f"\nAction taken: {action_name} (index: {action_idx})")
    print(f"Reward: {reward}, Done: {done}")

    # Display the new state
    print(env.display_state(next_state))

    # Show the updated grid
    plt.figure(figsize=(6, 6))
    plt.imshow(env.render())
    plt.title(f"After {action_name} action")
    plt.show()
