"""
Simple Linear Policy for Snake Game
This file implements a simple linear policy for the Snake game environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from snake_v2 import Snake


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(
        x - np.max(x))  # Subtracting the max value for numerical stability
    return e_x / e_x.sum()


def choose_action(state, weights):
    """
    Choose an action based on the current state and weights.

    Args:
        state: Current state observation
        weights: Weight matrix of shape (num_actions, num_features)

    Returns:
        action: Selected action
        prob: Probability distribution over actions
    """
    # Compute action probabilities using softmax(weights @ state)
    dot_product = np.dot(weights, state)
    # Sample action from probabilities
    prob = softmax(dot_product)
    action = np.random.choice(len(prob), p=prob)
    return action, prob


def run_episode(env, weights, max_steps=1000, render=False):
    """
    Run a single episode with the given weights.

    Args:
        env: Snake environment
        weights: Weight matrix
        max_steps: Maximum number of steps per episode
        render: Whether to render the environment

    Returns:
        total_reward: Total reward for the episode
        steps: Number of steps taken
    """
    state, _, done, _ = env.reset()
    total_reward = 0
    steps = 0

    action_map = {
        0: 'up',
        1: 'down',
        2: 'left',
        3: 'right',
    }

    while not done and steps < max_steps:
        action, _ = choose_action(state, weights)
        if render:
            print(f'Action: {action_map[action]}')
            plt.imshow(env.render())
            plt.pause(0.1)
            plt.clf()

        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, steps


def train_linear_policy(env, num_episodes=1000, learning_rate=0.01, gamma=0.99, exploration_prob=0.1, render_every=100):
    """
    Train a linear policy using simple policy gradient.

    Args:
        env: Snake environment
        num_episodes: Number of episodes to train for
        learning_rate: Learning rate
        gamma: Discount factor
        exploration_prob: Probability of taking a random action for exploration
        render_every: How often to render an episode

    Returns:
        weights: Trained weights
        all_rewards: List of rewards for each episode
    """
    # Initialize weights with zero mean and small standard deviation
    # This ensures no initial bias toward any action
    state_dim = len(env.reset()[0])
    action_dim = len(env.action_space)
    weights = np.random.normal(0, 0.1, size=(action_dim, state_dim))

    # Ensure all actions start with equal probability
    # by making the initial weights for each action similar
    weights = weights - np.mean(weights, axis=0)

    all_rewards = []

    for episode in range(num_episodes):
        # Reset environment
        state, _, done, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        # Run episode
        while not done:
            # Occasionally take random action for exploration
            if np.random.random() < exploration_prob:
                action = np.random.choice(env.action_space)
                # We don't need the probability for random actions
            else:
                action, _ = choose_action(state, weights)

            next_state, reward, done, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        # Compute discounted rewards
        discounted_rewards = np.zeros_like(episode_rewards)
        cumulative = 0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[i]
            discounted_rewards[i] = cumulative

        # Normalize rewards
        discounted_rewards = (
            discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)

        # Update weights
        for i in range(len(episode_rewards)):
            action_taken = episode_actions[i]
            state_i = episode_states[i]

            # Compute gradient
            dlog = -softmax(np.dot(weights, state_i))
            dlog[action_taken] += 1
            grad = np.outer(dlog, state_i)

            # Update weights
            weights += learning_rate * grad * discounted_rewards[i]

        # Track progress
        total_reward = sum(episode_rewards)
        all_rewards.append(total_reward)

        if episode % render_every == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            if render_every > 0:
                run_episode(env, weights, render=True)

    return weights, all_rewards


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train a linear policy for Snake game')
    parser.add_argument('--grid_size', type=int, default=21,
                        help='Size of the grid (default: 21)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train for (default: 100)')
    parser.add_argument('--render_every', type=int, default=25,
                        help='Render every N episodes during training (default: 25)')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--exploration', type=float, default=0.1,
                        help='Exploration probability (default: 0.1)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting of learning curve')
    args = parser.parse_args()

    # Create environment
    env = Snake(grid_size=args.grid_size)

    # Train the policy
    print(
        f"Training linear policy on {args.grid_size}x{args.grid_size} grid for {args.episodes} episodes...")
    weights, rewards = train_linear_policy(
        env,
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        exploration_prob=args.exploration,
        render_every=args.render_every
    )

    # Test the trained policy
    print("\nTesting trained policy...")
    total_reward, steps = run_episode(env, weights, render=True)
    print(f"Test episode - Total Reward: {total_reward}, Steps: {steps}")

    # Plot learning curve
    if not args.no_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title(
            f'Learning Curve - Linear Policy (Grid: {args.grid_size}x{args.grid_size}, Episodes: {args.episodes})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()
