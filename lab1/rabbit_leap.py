from collections import deque

def is_valid(state):
    empty = list(state).index(-1)
    if(state==(1,1,1,-1,0,0,0)):
        return True
    if(empty == 0 and state[empty+1]==0  and state[empty+2] == 0):
        return False
    if(empty == 6 and state[empty-1]==1 and state[empty-2] == 1):
        return False 
    if(empty == 1 and state[empty-1]==1 and state[empty+1]==0 and state[empty+2]==0):
        return False
    if(empty == 5 and state[empty+1]==0 and state[empty-1]==1 and state[empty-2]==1):
        return False
    if(state[empty-1]==1 and state[empty-2]==1 and state[empty+1]==0 and state[empty+2]==0):
        return False    
    return True

def swap(state, i, j):
    new_state = list(state)
    temp = new_state[i]
    new_state[i] = new_state[j]
    new_state[j] = temp
    return tuple(new_state)

def get_successors(state):
    successors = []
    empty = list(state).index(-1)
    moves = [-2,-1,1,2]
    for move in moves:
        if(empty + move >= 0 and empty + move < 7):
            if(move>0 and list(state)[move+empty]==1):
                new_state = swap(state, empty, empty + move)
                if is_valid(new_state):
                    successors.append(new_state)
            if(move<0 and list(state)[move+empty]==0):
                new_state = swap(state, empty, empty + move)
                if(state==(1,1,1,0,-1,0,0)):
                    print(new_state)
                if is_valid(new_state):
                    successors.append(new_state)
    return successors

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    count=0
    max_size=0
    while queue:
        size=len(queue)
        max_size=max(size,max_size)
        (state, path) = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        count+=1
        if state == goal_state:
            print(f"Total Number Of Nodes Visited: {count}")
            print(f"Max Size Of queue at a point was: {max_size}")
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (0,0,0,-1,1,1,1)
goal_state = (1,1,1,-1,0, 0, 0)

solution = bfs(start_state, goal_state)
if solution:
    print("Solution found:")
    print(f"Number Of nodes in solution: {len(solution)}")
    for step in solution:
        print(step)
else:
    print("No solution found.")
