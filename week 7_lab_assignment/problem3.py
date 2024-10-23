import numpy as np

class NonStationaryBandit:
    def __init__(self, k=10, std_walk=0.01, init_mean=0):  # Corrected __init__ method
        self.k = k  # Number of arms
        self.means = np.full(k, float(init_mean))  # Initialize all means to the same value
        self.std_walk = std_walk  # Standard deviation of the random walk
        self.time_step = 0  # Track time steps
        
        # Estimated rewards and counts for each arm
        self.q_estimates = np.zeros(k)  # Estimated values of each arm
        self.action_counts = np.zeros(k)  # Count of actions taken for each arm

    def update_means(self):
        self.means += np.random.normal(0, self.std_walk, self.k)

    def pull(self, action):
        # Update means before pulling the arm (non-stationary bandit)
        self.update_means()

        # Return the reward sampled from a normal distribution centered at the current mean
        reward = self.means[action]
        return reward

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            # Exploration: Choose a random action
            action = np.random.randint(0, self.k)
        else:
            # Exploitation: Choose the action with the highest estimated value
            action = np.argmax(self.q_estimates)
        
        return action

    def update_estimates(self, action, reward):
        # Update counts and estimates
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

# Simulation
def simulate_bandit(n_steps=1000, epsilon=0.1):
    # Initialize the 10-armed bandit
    bandit = NonStationaryBandit()
    
    # Track rewards
    rewards = np.zeros(n_steps)

    for t in range(n_steps):
        # Select action using epsilon-greedy strategy
        action = bandit.select_action(epsilon)

        # Pull the chosen arm and get the reward
        reward = bandit.pull(action)
        rewards[t] = reward

        # Update estimates based on the received reward
        bandit.update_estimates(action, reward)

    # Calculate average reward
    average_reward = np.mean(rewards)
    
    return rewards, average_reward

# Run the simulation
n_steps = 10000
epsilon = 0.1  # Exploration probability
rewards, average_reward = simulate_bandit(n_steps, epsilon)

# Print average reward
print(f"Average reward over {n_steps} steps: {average_reward}")

# Example plot of rewards over time
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Rewards over time for a 10-armed non-stationary bandit')
plt.show()
