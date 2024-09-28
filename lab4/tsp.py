import numpy as np
import random

tourist_locations = [
    "Jaipur",
    "Jodhpur",
    "Udaipur",
    "Bikaner",
    "Ajmer",
    "Mount Abu",
    "Pushkar",
    "Jaisalmer",
    "Bundi",
    "Alwar",
    "Chittorgarh",
    "Kota",
    "Ranthambore",
    "Bharatpur",
    "Sawai Madhopur",
    "Pali",
    "Nagaur",
    "Bhilwara",
    "Barmer",
    "Jalore",
]

np.random.seed(42)
num_locations = len(tourist_locations)
dist_matrix = np.random.randint(50, 500, size=(num_locations, num_locations))
np.fill_diagonal(dist_matrix, 0)


def get_cost(tour, dist_matrix):
    """Calculate the total distance of the given tour."""
    if len(tour) <= 1:
        return 0
    return sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


def get_successor(tour):
    """Generate a neighboring solution by swapping two cities in the tour."""
    if len(tour) < 2:
        return tour
    successor = tour.copy()
    idx1, idx2 = random.sample(range(len(successor)), 2)
    successor[idx1], successor[idx2] = successor[idx2], successor[idx1]
    return successor


def hill_climb_expand(tour, remaining_cities, dist_matrix):
    """Expand the current tour by adding the best city from the remaining cities using Hill Climbing."""
    best_tour = None
    best_cost = float("inf")

    for city in remaining_cities:
        new_tour = tour + [city]
        new_cost = get_cost(new_tour, dist_matrix)
        if new_cost < best_cost:
            best_tour = new_tour
            best_cost = new_cost

    return best_tour, best_cost


# Simulated Annealing algorithm
def simulated_annealing(
    tour, dist_matrix, initial_temperature, cooling_rate, max_iterations
):
    """Apply Simulated Annealing to escape local minima and find a better solution."""
    current_tour = tour
    current_cost = get_cost(current_tour, dist_matrix)

    best_tour = current_tour.copy()
    best_cost = current_cost

    temperature = initial_temperature

    for iteration in range(max_iterations):
        successor = get_successor(current_tour)
        successor_cost = get_cost(successor, dist_matrix)

        delta_cost = successor_cost - current_cost
        acceptance_probability = (
            np.exp(-delta_cost / temperature) if delta_cost > 0 else 1.0
        )

        if acceptance_probability > random.random():
            current_tour = successor
            current_cost = successor_cost

        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost

        temperature *= cooling_rate

    return best_tour, best_cost


def build_and_optimize_tour(
    dist_matrix, initial_temperature, cooling_rate, max_iterations
):
    initial_tour = []
    remaining_cities = set(range(len(dist_matrix)))

    while remaining_cities:
        initial_tour, _ = hill_climb_expand(initial_tour, remaining_cities, dist_matrix)
        remaining_cities -= set(initial_tour)

    best_tour, best_cost = simulated_annealing(
        initial_tour, dist_matrix, initial_temperature, cooling_rate, max_iterations
    )

    return best_tour, best_cost


initial_temperature = 40000
cooling_rate = 0.9998
max_iterations = 20000

final_tour, final_cost = build_and_optimize_tour(
    dist_matrix, initial_temperature, cooling_rate, max_iterations
)

print("Final Optimized Tour:")
for idx in final_tour:
    print(tourist_locations[idx], end=" -> ")
print(tourist_locations[final_tour[0]])
print(f"Total Cost of the Best Tour: {final_cost} km")
