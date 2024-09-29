import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import copy
from tqdm import tqdm

from collections import deque


def load_octave_column_matrix(file_path):
    matrix = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip the header (first 5 lines)
    matrix_lines = lines[5:]

    # Process the matrix data, skipping any empty or non-numeric lines
    for line in matrix_lines:
        line = line.strip()
        if line:  # Check if the line is not empty
            try:
                matrix.append(int(line))  # Convert to integer and append
            except ValueError:
                print(f"Skipping invalid line: {line}")  # Handle invalid lines

    # Convert the list to a NumPy array and reshape it into 512x512
    matrix = np.array(matrix)

    # Check if the total data length is correct (512*512 = 262144)
    if matrix.size != 512 * 512:
        raise ValueError(f"Expected 262144 elements, but got {matrix.size} elements.")

    reshaped_matrix = matrix.reshape((512, 512))  # Reshape into 512x512 matrix

    return reshaped_matrix


def create_patch(image):
    patch_size = 128

    # Number of patches along each dimension
    num_patches = image.shape[0] // patch_size

    # List to store patches
    patches = {}
    state_mat = []
    z = 0
    # Loop to extract 128x128 patches
    for i in range(num_patches):
        temp = []
        for j in range(num_patches):
            temp.append(z)

            # Extract patch using array slicing
            patch = image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            patches[z] = patch
            z = z + 1

        state_mat.append(temp)

    # 'patches' will contain 16 patches of size 128x128

    return patches, state_mat


def reconstruct_image(patches, grid):
    """
    Reconstructs a full image from patches using a grid.

    :param patches: List of image patches as NumPy arrays.
    :param grid: 2D list where each element is an index pointing to a patch in `patches`.
    :return: Reconstructed image.
    """
    patch_height, patch_width = patches[0].shape[:2]
    grid_height = len(grid)
    grid_width = len(grid[0])

    # Initialize an empty array for the full image
    full_image = np.zeros(
        (grid_height * patch_height, grid_width * patch_width), dtype=np.uint8
    )

    for i, row in enumerate(grid):
        for j, patch_index in enumerate(row):
            # Place each patch in its corresponding position in the full image
            full_image[
                i * patch_height : (i + 1) * patch_height,
                j * patch_width : (j + 1) * patch_width,
            ] = patches[patch_index]

    return full_image


# Initialize a 4x4 grid with zeros


def get_score(array1, array2):
    diffrance = 0
    for i in range(len(array1)):
        diffrance = diffrance + abs(array1[i] - array2[i])
    return diffrance


def get_value(value, paren_patch, patch, directions):
    max_score = float("inf")
    max_vale = -1
    paren_patch = np.array(paren_patch)
    for i in value:
        score = 0
        child_patch = np.array(patch[i])

        if directions == (0, 1):
            for j in range(128):
                score += abs(paren_patch[j][127] - child_patch[j][0])
        if directions == (0, -1):
            for j in range(128):
                score += abs(paren_patch[j][0] - child_patch[j][127])
        if directions == (1, 0):
            for j in range(128):
                score += abs(paren_patch[127][j] - child_patch[0][j])
        if directions == (-1, 0):
            for j in range(128):
                score += abs(paren_patch[0][j] - child_patch[127][j])

        if score < max_score:
            max_score = score
            max_vale = i

    return max_vale


def bfs_fill(grid, patch, value):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque([(0, 0)])
    visited = set()
    visited.add((0, 0))

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))
                current_val = get_value(value, patch[grid[x][y]], patch, (dx, dy))
                grid[nx][ny] = current_val
                value.remove(current_val)


def image(image):
    plt.imshow(image, cmap="gray")
    plt.title("512x512 Matrix Visualization")
    plt.colorbar()  # Optional: shows a color bar indicating intensity levels
    plt.show()


def get_neighbors(i, j, grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])

    # Check Up
    if i - 1 >= 0:
        neighbors.append((i - 1, j))

    # Check Down
    if i + 1 < rows:
        neighbors.append((i + 1, j))

    # Check Left
    if j - 1 >= 0:
        neighbors.append((i, j - 1))

    # Check Right
    if j + 1 < cols:
        neighbors.append((i, j + 1))

    return neighbors


def value_function(grid, patches):
    score = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            neighbors = get_neighbors(i, j, grid)

            for k in neighbors:
                if k[1] == j + 1:
                    img1 = patches[grid[k[0]][k[1]]]
                    img2 = patches[grid[i][j]]

                    for m in range(128):
                        score += abs(img2[127][m] - img1[0][m])

                if k[1] == j - 1:
                    img1 = patches[grid[k[0]][k[1]]]
                    img2 = patches[grid[i][j]]

                    for m in range(128):
                        score += abs(img1[127][m] - img2[0][m])

                if k[0] == i + 1:
                    img1 = patches[grid[k[0]][k[1]]]
                    img2 = patches[grid[i][j]]

                    for m in range(128):
                        score += abs(img2[m][127] - img1[m][0])

                if k[0] == i - 1:
                    img1 = patches[grid[k[0]][k[1]]]
                    img2 = patches[grid[i][j]]

                    for m in range(128):
                        score += abs(img1[m][127] - img2[m][0])

    return np.sqrt(score)


def calculate_gradients(image, threshold=100):
    """
    Calculate the edge gradients in x and y directions using Sobel filters.

    :param image: Input image.
    :return: Gradients in x and y directions and their sum.
    """
    gray = image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_x[magnitude < threshold] = 0
    grad_y[magnitude < threshold] = 0
    grad_sum = np.sqrt(np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y)))

    return grad_sum


def simulated_annealing(grid, patches, value):
    current_grid = copy.deepcopy(grid)
    best_grid = copy.deepcopy(grid)
    current_score = value_function(current_grid, patches)
    best_score = current_score

    initial_temp = 10
    final_temp = 1
    alpha = 0.995  # Cooling rate
    temp = initial_temp

    while temp > final_temp:
        x1, y1 = random.randint(0, 3), random.randint(0, 3)
        x2, y2 = random.randint(0, 3), random.randint(0, 3)

        current_grid[x1][y1], current_grid[x2][y2] = (
            current_grid[x2][y2],
            current_grid[x1][y1],
        )

        new_score = value_function(current_grid, patches)

        if new_score < current_score or random.uniform(0, 1) < math.exp(
            (current_score - new_score) / temp
        ):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_grid = copy.deepcopy(current_grid)
        else:
            current_grid[x1][y1], current_grid[x2][y2] = (
                current_grid[x2][y2],
                current_grid[x1][y1],
            )

        temp *= alpha

    return best_grid, best_score


if __name__ == "__main__":
    images = load_octave_column_matrix("jigsaww.mat")
    print(images.shape)  # Should print (512, 512)

    images = images.T

    patch, state_mat = create_patch(images)
    final_grid = None
    final_score = float("inf")
    for i in range(16):
        grid = [[-1 for _ in range(4)] for _ in range(4)]
        grid[0][0] = i
        value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        value.remove(i)
        bfs_fill(grid, patch, value)
        new_grid = copy.deepcopy(grid)
        new_img = reconstruct_image(patch, new_grid)
        score = calculate_gradients(new_img)
        final_grid_1, new_score = simulated_annealing(new_grid, patch, score)
        if score < final_score:
            final_grid = grid
            final_score = score

        best_img = reconstruct_image(patch, final_grid)
        new_score = calculate_gradients(best_img)
        if new_score < final_score:
            final_grid = final_grid_1
            final_score = new_score
    best_img = reconstruct_image(patch, final_grid)
    image(best_img)
