import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        # Ensure no self-connection
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=10):
        state = np.array(pattern)
        for _ in range(max_iterations):
            state = np.sign(np.dot(self.weights, state))
        return state

    def capacity(self):
        return int(0.15 * self.num_neurons)

num_neurons = 10 * 10
hopfield = HopfieldNetwork(num_neurons)

num_patterns = hopfield.capacity()
patterns = [np.random.choice([-1, 1], size=(num_neurons,)) for _ in range(num_patterns)]
hopfield.train(patterns)

test_pattern = patterns[0].copy()
noisy_pattern = test_pattern.copy()
noise_indices = np.random.choice(num_neurons, size=num_neurons // 10, replace=False)
noisy_pattern[noise_indices] *= -1

recalled_pattern = hopfield.recall(noisy_pattern)

print("Original Pattern:", test_pattern.reshape(10, 10))
print("Noisy Pattern:", noisy_pattern.reshape(10, 10))
print("Recalled Pattern:", recalled_pattern.reshape(10, 10))

print("Theoretical Capacity of Hopfield Network:", hopfield.capacity(), "patterns")
