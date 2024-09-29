import heapq

class SolitaireNode:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state  # 7x7 grid
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost estimate to goal
        self.f = w1 * g + w2 * h  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def get_possible_moves(state):
    """
    Get all possible valid moves in the current state of the board.
    Each move is represented by a tuple (start_row, start_col, end_row, end_col).
    """
    moves = []
    
    # Define the directions for moving: (row_change, col_change)
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # Up, Down, Left, Right
    jump_over = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # The jumped over marble position
    
    # Loop through every position in the 7x7 board
    for r in range(7):
        for c in range(7):
            # Check if the current position has a marble ('O')
            if state[r][c] == 'O':
                # Try all four possible directions
                for (dr, dc), (jr, jc) in zip(directions, jump_over):
                    end_r = r + dr
                    end_c = c + dc
                    jump_r = r + jr
                    jump_c = c + jc
                    
                    # Check if the move stays within the board bounds
                    if 0 <= end_r < 7 and 0 <= end_c < 7:
                        # Check if there is a marble at the jump position and empty space at the destination
                        if state[jump_r][jump_c] == 'O' and state[end_r][end_c] == '0':
                            # Valid move: (start_row, start_col, end_row, end_col)
                            moves.append((r, c, end_r, end_c))
    
    return moves
def apply_move(state, move):
    # Deep copy the current state to avoid modifying the original state
    new_state = [row[:] for row in state]
    
    # Extract move details
    start_row, start_col, end_row, end_col = move
    
    # Calculate the position of the marble being jumped over
    jump_row = (start_row + end_row) // 2
    jump_col = (start_col + end_col) // 2
    
    # Apply the move: 
    # 1. Move the marble from start to end position
    new_state[end_row][end_col] = 'O'
    
    # 2. Empty the start position
    new_state[start_row][start_col] = '0'
    
    # 3. Remove the jumped-over marble
    new_state[jump_row][jump_col] = '0'
    
    return new_state


def heuristic_1(state):
    return sum(row.count('O') for row in state)

def heuristic_2(state):
    center = (3, 3)
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                total_distance += abs(r - center[0]) + abs(c - center[1])
    return total_distance

def best_first_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.h, start_node))
    visited = set()
    max_size=0
    while open_list:
        _, node = heapq.heappop(open_list)
        if len(open_list)>max_size:
            max_size=len(open_list)
        if tuple(map(tuple, node.state)) in visited:
            continue
        visited.add(tuple(map(tuple, node.state)))

        if heuristic_func(node.state) == 0:  # Goal: Only one marble left
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("max number of nodes in queue is: ", max_size)    
            print("number of nodes visited is: ", len(visited))
            print("number of nodes in the path is: ", len(path))
            return path[::-1]

        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, node, h=h)
            heapq.heappush(open_list, (new_node.h, new_node))

    return None
def a_star_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    max_size=0
    while open_list:
        _, node = heapq.heappop(open_list)
        if len(open_list)>max_size:
            max_size=len(open_list)
        if tuple(map(tuple, node.state)) in visited:
            continue
        visited.add(tuple(map(tuple, node.state)))

        if heuristic_func(node.state) == 0:  # Goal: Only one marble left
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("max number of nodes in queue is: ", max_size)
            print("number of nodes visited is: ", len(visited))
            print("number of nodes in the path is: ", len(path))
            return path[::-1]

        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            g = node.g + 1  # Increment path cost
            h = heuristic_func(new_state)
            f = g + h
            new_node = SolitaireNode(new_state, node, g=g, h=h)
            heapq.heappush(open_list, (new_node.f, new_node))

    return None

start_state = [
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', '0', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['-', '-', 'O', 'O', 'O', '-', '-'],
    ['-', '-', 'O', 'O', 'O', '-', '-']
]

moves = get_possible_moves(start_state)

# Print the possible moves
for move in moves:
    print(f"Start: {move[0], move[1]} -> End: {move[2], move[3]}")
print("Best first search results are: ")
best_first_search(start_state, heuristic_2)
print("A* search results are: ")
a_star_search(start_state, heuristic_2)
