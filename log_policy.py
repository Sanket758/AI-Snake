"""
Log Policy for Snake Game
This file implements a policy gradient method with log probabilities for the Snake game environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from snake_v2 import Snake
from tqdm import tqdm


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


def compute_discounted_rewards(rewards, gamma=0.99):
    """
    Compute discounted rewards for an episode.

    Args:
        rewards: List of rewards for an episode
        gamma: Discount factor

    Returns:
        discounted: Normalized discounted rewards
    """
    discounted = np.zeros_like(rewards)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted[i] = cumulative

    # Normalize rewards
    mean = np.mean(discounted)
    std = np.std(discounted)
    discounted = (discounted - mean) / (std + 1e-10)

    return discounted


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


def train_log_policy(env, num_episodes=1000, learning_rate=0.01, gamma=0.99, exploration_prob=0.1, render_every=100):
    """
    Train a policy using policy gradient with log probabilities.

    Args:
        env: Snake environment
        num_episodes: Number of episodes to train for
        learning_rate: Learning rate
        gamma: Discount factor
        exploration_prob: Probability of taking a random action
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

    for episode in tqdm(range(num_episodes), desc="Training Log Policy"):
        # Reset environment
        state, _, done, _ = env.reset()
        episode_states = [state]
        episode_actions = []
        episode_probs = []
        episode_rewards = []
        episode_gradients = []

        # Run episode
        while not done:
            # Occasionally take random action for exploration
            if np.random.random() < exploration_prob:
                action = np.random.choice(env.action_space)
                prob = np.ones(action_dim) / action_dim
            else:
                action, prob = choose_action(state, weights)

            next_state, reward, done, _ = env.step(action)

            episode_states.append(next_state)
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_probs.append(prob)

            state = next_state

        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)

        # Update weights using log policy gradient
        for i in range(len(episode_rewards)):
            action_taken = episode_actions[i]
            state_i = episode_states[i]

            # Compute gradient of log policy
            dlog = -episode_probs[i]  # -prob for all actions
            dlog[action_taken] += 1   # add 1 for the action taken

            # Compute gradient
            grad = np.outer(dlog, state_i)
            episode_gradients.append(grad)

            # Update weights
            weights += learning_rate * grad * discounted_rewards[i]

        # Track progress
        total_reward = sum(episode_rewards)
        all_rewards.append(total_reward)

        if episode % render_every == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            if render_every > 0 and episode > 0:
                run_episode(env, weights, render=True)

    return weights, all_rewards


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train a log policy for Snake game')
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
        f"Training log policy on {args.grid_size}x{args.grid_size} grid for {args.episodes} episodes...")
    weights, rewards = train_log_policy(
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
            f'Learning Curve - Log Policy (Grid: {args.grid_size}x{args.grid_size}, Episodes: {args.episodes})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()
