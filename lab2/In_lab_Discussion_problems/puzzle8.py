import numpy as np
import sys
from time import time


class Node:
    def __init__(self, parent, state, pcost, hcost):
        self.parent = parent
        self.state = state
        self.pcost = pcost
        self.hcost = hcost
        self.cost = pcost + hcost

    def __hash__(self):
        return hash("".join(self.state.flatten()))

    def __str__(self):
        return str(self.state)

    def __eq__(self, other):
        return hash("".join(self.state.flatten())) == hash(
            "".join(other.state.flatten())
        )

    def __ne__(self, other):
        return hash("".join(self.state.flatten())) != hash(
            "".join(other.state.flatten())
        )


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, node):
        self.queue.append(node)

    def pop(self):
        next_state = None
        state_cost = 10**18
        index = -1

        for i in range(len(self.queue)):
            if self.queue[i].cost < state_cost:
                state_cost = self.queue[i].cost
                index = i

        return self.queue.pop(index)

    def is_empty(self):
        return len(self.queue) == 0

    def __str__(self):
        l = []
        for i in self.queue:
            l.append(i.state)

        return str(l)

    def __len__(self):
        return len(self.queue)


class Environment:
    def __init__(self, depth=None, goal_state=None):
        self.actions = [1, 2, 3, 4]  # 1 - Up, 2 - Down, 3 - Right, 4 - Left
        self.goal_state = goal_state
        self.depth = depth
        self.start_state = self.generate_start_state()

    def generate_start_state(self):
        past_state = self.goal_state
        i = 0
        while i != self.depth:
            new_states = self.get_next_states(past_state)
            choice = np.random.randint(low=0, high=len(new_states))

            if np.array_equal(new_states[choice], past_state):
                continue

            past_state = new_states[choice]
            i += 1

        return past_state

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state

    def get_next_states(self, state):
        space = (0, 0)
        for i in range(3):
            for j in range(3):
                if state[i, j] == "_":
                    space = (i, j)
                    break

        new_states = []

        if space[0] > 0:  # Move Up
            new_state = np.copy(state)

            val = new_state[space[0], space[1]]
            new_state[space[0], space[1]] = new_state[space[0] - 1, space[1]]
            new_state[space[0] - 1, space[1]] = val

            new_states.append(new_state)

        if space[0] < 2:  # Move down
            new_state = np.copy(state)

            val = new_state[space[0], space[1]]
            new_state[space[0], space[1]] = new_state[space[0] + 1, space[1]]
            new_state[space[0] + 1, space[1]] = val

            new_states.append(new_state)

        if space[1] < 2:  # Move right
            new_state = np.copy(state)

            val = new_state[space[0], space[1]]
            new_state[space[0], space[1]] = new_state[space[0], space[1] + 1]
            new_state[space[0], space[1] + 1] = val

            new_states.append(new_state)

        if space[1] > 0:  # Move Left
            new_state = np.copy(state)

            val = new_state[space[0], space[1]]
            new_state[space[0], space[1]] = new_state[space[0], space[1] - 1]
            new_state[space[0], space[1] - 1] = val

            new_states.append(new_state)

        return new_states

    def reached_goal(self, state):
        for i in range(3):
            for j in range(3):
                if state[i, j] != self.goal_state[i, j]:
                    return False

        return True


class Agent:
    def __init__(self, env, heuristic):
        self.frontier = PriorityQueue()
        self.explored = dict()
        self.start_state = env.get_start_state()
        self.goal_state = env.get_goal_state()
        self.env = env
        self.goal_node = None
        self.heuristic = heuristic

    def run(self):
        init_node = Node(parent=None, state=self.start_state, pcost=0, hcost=0)
        self.frontier.push(init_node)
        steps = 0
        while not self.frontier.is_empty():
            curr_node = self.frontier.pop()
            # print(curr_node.cost)
            next_states = self.env.get_next_states(curr_node.state)

            if hash(curr_node) in self.explored:
                continue

            self.explored[hash(curr_node)] = curr_node

            if self.env.reached_goal(curr_node.state):
                # print("Reached goal!")
                self.goal_node = curr_node
                break
            goal_state = self.env.get_goal_state()

            l = []
            for state in next_states:
                hcost = self.heuristic(state, goal_state)
                node = Node(
                    parent=curr_node,
                    state=state,
                    pcost=curr_node.pcost + 1,
                    hcost=hcost,
                )
                self.frontier.push(node)
            steps += 1

        return steps, self.soln_depth()

    def soln_depth(self):
        node = self.goal_node
        count = 0
        while node is not None:
            node = node.parent
            count += 1

        return count

    def print_nodes(self):
        node = self.goal_node
        l = []
        while node is not None:
            l.append(node)
            node = node.parent

        step = 1
        for node in l[::-1]:
            print("Step: ", step)
            print(node)
            step += 1

    def get_memory(self):
        mem = len(self.frontier) * 56 + len(self.explored) * 56
        return mem


def heuristic0(curr_state, goal_state):
    return 0


def heuristic1(curr_state, goal_state):
    count = 0
    for i in range(3):
        for j in range(3):
            if curr_state[i, j] != goal_state[i, j]:
                count += 1

    return count


def heuristic2(curr_state, goal_state):
    dist = 0

    for i in range(3):
        for j in range(3):
            ele = curr_state[i, j]
            goal_i, goal_j = np.where(goal_state == ele)
            d = abs(goal_i[0] - i) + abs(goal_j[0] - j)
            dist += d

    return dist


depth = 500
goal_state = np.array([[1, 2, 3], [8, "_", 4], [7, 6, 5]])
env = Environment(depth, goal_state)
print("Start State: ")
print(env.get_start_state())
print("Goal State: ")
print(goal_state)

agent = Agent(env=env, heuristic=heuristic2)
agent.run()

depths = np.arange(0, 501, 50)
goal_state = np.array([[1, 2, 3], [8, "_", 4], [7, 6, 5]])
times_taken = {}
mems = {}
for depth in depths:
    time_taken = 0
    mem = 0
    for i in range(50):
        env = Environment(depth=depth, goal_state=goal_state)
        agent = Agent(env=env, heuristic=heuristic2)
        start_time = time()
        agent.run()
        end_time = time()
        time_taken += end_time - start_time
        mem += agent.get_memory()

    time_taken /= 50
    mem = mem / 50
    times_taken[depth] = time_taken
    mems[depth] = mem
    print(depth, time_taken, mem)

