import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_cities = 10 
gamma = 1000 
city_coordinates = np.random.rand(n_cities, 2) * 100
d = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        d[i, j] = np.linalg.norm(city_coordinates[i] - city_coordinates[j])

plt.figure(figsize=(8, 6))
sns.heatmap(d, annot=True, cmap='Blues', xticklabels=[f"City {i+1}" for i in range(n_cities)],
            yticklabels=[f"City {i+1}" for i in range(n_cities)], cbar=True)
plt.title("Initial Distance Matrix Heatmap")
plt.xlabel("Cities")
plt.ylabel("Cities")
plt.show()

n_units = n_cities ** 2  
W = np.zeros((n_units, n_units))  
biases = -gamma / 2 * np.ones(n_units)
np.random.seed(42)
def get_index(city, position, n_cities):
    return city * n_cities + position

for i in range(n_cities):
    for k in range(n_cities):
        for j in range(n_cities):
            for l in range(n_cities):
                if k == (l - 1) % n_cities: 
                    W[get_index(i, k, n_cities), get_index(j, l, n_cities)] -= d[i, j]

for i in range(n_cities):
    for k in range(n_cities):
        for l in range(n_cities):
            if k != l:  
                W[get_index(i, k, n_cities), get_index(i, l, n_cities)] -= gamma
        for j in range(n_cities):
            if i != j:
                W[get_index(i, k, n_cities), get_index(j, k, n_cities)] -= gamma

np.fill_diagonal(W, 0)

def get_index(i, k, n_cities):
    # Placeholder function. Define your method of computing the index for the weight matrix.
    return i * n_cities + k

def calculate_total_distance(tour, d):
    # Function to calculate the total distance of a tour.
    total_distance = 0
    for i in range(len(tour) - 1):
        city_a = tour[i]
        city_b = tour[i + 1]
        total_distance += d[city_a, city_b]
    return total_distance

# Variables for storing the best tour
best_tour = None
best_distance = float('inf')

# Repeat the process 100 times
for repeat in range(100):
    # Initialize the state
    state = np.random.choice([0,1], size=n_units)
    state = state.reshape(n_cities, n_cities)
    
    # Create a random initial state
    for i in range(n_cities):
        state[i, :] = 0
        state[i, np.random.randint(0, n_cities)] = 1

    for j in range(n_cities):
        state[:, j] = 0
        state[np.random.randint(0, n_cities), j] = 1

    # Threshold for decision making
    threshold = -(gamma/2)

    # Iterations to update the state
    for iteration in range(1000):
        prev_state = state.copy()
        for i in range(n_cities):
            for k in range(n_cities):
                idx = get_index(i, k, n_cities)
                input_sum = np.dot(W[idx, :], state.flatten()) + biases[idx]
                state[i, k] = 1 if input_sum >= 1.5 * threshold else 0

        # Check for convergence
        if np.array_equal(state, prev_state):
            break
    
    # Extract the tour from the state
    tour = []
    for step in range(n_cities):
        for city in range(n_cities):
            if state[city, step] == 1:
                tour.append(city)
                break

    tour.append(tour[0])  # Close the loop by returning to the first city

    # Calculate the total distance of the current tour
    total_distance = calculate_total_distance(tour, d)

    # Update the best tour if the current one is better
    if total_distance < best_distance:
        best_distance = total_distance
        best_tour = tour

# After 100 repetitions, print the best tour
print("Best Tour found:", best_tour)
print("Best Total Distance:", best_distance)

tour = best_tour
print(f"Tour {tour}")
plt.figure(figsize=(8, 8))
for i, (x, y) in enumerate(city_coordinates):
    plt.scatter(x, y, color="red", s=100, label="City" if i == 0 else "")
    plt.text(x + 2, y + 2, f"City {i+1}", fontsize=12)

for i in range(len(tour) - 1):
    city_a = city_coordinates[tour[i]]
    city_b = city_coordinates[tour[i + 1]]
    plt.plot([city_a[0], city_b[0]], [city_a[1], city_b[1]], "b-", linewidth=2, label="Path" if i == 0 else "")

plt.title("Traveling Salesman Problem Solution", fontsize=16)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid()
plt.show()


def greedy_tsp(distance_matrix):
    n = len(distance_matrix)
    visited = [False] * n 
    tour = [0]
    visited[0] = True 

    total_distance = 0 
    current_city = 0 

    for _ in range(n - 1):
        nearest_city = None
        nearest_distance = 1000000

        # Find the nearest unvisited city
        for i in range(n):
            if not visited[i] and distance_matrix[current_city][i] < nearest_distance:
                nearest_city = i
                nearest_distance = distance_matrix[current_city][i]

        tour.append(nearest_city)
        total_distance += nearest_distance
        visited[nearest_city] = True
        current_city = nearest_city 

    total_distance += distance_matrix[current_city][tour[0]]
    tour.append(tour[0])

    return tour, total_distance

# Solve TSP using Greedy algorithm
tour, total_distance = greedy_tsp(d)
print("Greedy TSP Solution:",total_distance)
