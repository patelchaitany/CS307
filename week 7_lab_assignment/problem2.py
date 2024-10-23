import numpy as np


# Define binary_bandit_a and binary_bandit_b
def binary_bandit_a(action):
    p = [0.1, 0.2]  # Success probabilities for actions 1 and 2
    if np.random.rand() < p[action - 1]:
        return 1  # Success
    else:
        return 0  # Failure


def binary_bandit_b(action):
    p = [0.8, 0.9]  # Success probabilities for actions 1 and 2
    if np.random.rand() < p[action - 1]:
        return 1  # Success
    else:
        return 0  # Failure


# Epsilon-greedy algorithm
def epsilon_greedy_bandit(epsilon, num_iterations, bandit_function):
    Q = [0, 0]  # Action-value estimates for actions 1 and 2
    N = [0, 0]  # Number of times each action has been taken
    total_reward = 0

    for t in range(num_iterations):
        # Exploration vs Exploitation
        if np.random.rand() < epsilon:
            action = np.random.choice([1, 2])  # Explore: Random action (1 or 2)
        else:
            action = np.argmax(Q) + 1  # Exploit: Select the best current action

        # Get reward from the bandit
        reward = bandit_function(action)

        # Update action-value estimate
        N[action - 1] += 1
        Q[action - 1] += (reward - Q[action - 1]) / N[action - 1]

        # Accumulate total reward
        total_reward += reward

        # Optional: Print status

    # Display final results
    print(f"\nFinal Action-Value Estimates: Q(1) = {Q[0]:.2f}, Q(2) = {Q[1]:.2f}")


# Run the epsilon-greedy algorithm for binary_bandit_a
epsilon_greedy_bandit(0.1, 10000, binary_bandit_a)

# Run the epsilon-greedy algorithm for binary_bandit_b
epsilon_greedy_bandit(0.1, 10000, binary_bandit_b)
