from lab3_gen import generate_k_sat_problem
import random
#definition of class node which which will have definition of node
class Node:
    def __init__(self,state):
        self.state=state
#this is heuristic function 1 will calculate the hueristic value of the node with respect to clause    
def heuristic_value_1(clause,node):
    count=0
    for curr_clause in clause:
        for i in curr_clause:
            if i>0 and node.state[i-1]==1:
                count+=1
                break
            if i<0 and node.state[abs(i)-1]==0:
                count+=1
                break
    return count
#this is heuristic function 2 which will calculate the heuristic value of the node with respect to clause
def heuristic_value_2(clause, node):
    state = node.state
    count = 0
    for curr_clause in clause:
        for literal in curr_clause:
            if state[abs(literal) - 1] == 1:
                count += 1
    return count
#this function will check if the node is the solution or not
def check(clause, node):
    count=0
    for curr_clause in clause:
        for i in curr_clause:
            if i>0 and node.state[i-1]==1:
                count+=1
                break
            if i<0 and node.state[abs(i)-1]==0:
                count+=1
                break
    if count==len(clause):
        return True
    return False
#this function generates all possible sucessors for hill-climb algorithm
def gen_1(node,clause):
    max=-1
    max_node=node
    i=random.randint(0,len(node.state)-1)
    for i in range(len(node.state)):
        temp=node.state.copy()
        if temp[i]==0:
            temp[i]=1
        elif temp[i]==1:
            temp[i]=0
        new_node= Node(state=temp)
        val=heuristic_value_2(clause,new_node)
        if val>max:
            max=val
            max_node=new_node
    if max_node.state==node.state:
        return None
    return max_node
def gen_2(node, clause, num_neighbors=10):
    max_value = -1
    max_node = node

    # Generate `num_neighbors` random successors by flipping one or two bits each time
    for _ in range(num_neighbors):
        temp = node.state.copy()
        
        # Randomly choose between flipping one or two bits
        num_bits_to_flip = random.choice([1, 2])

        if num_bits_to_flip == 1:
            i = random.randint(0, len(node.state) - 1)
            temp[i] = 1 - temp[i]  # Flip one bit

        elif num_bits_to_flip == 2:
            i, j = random.sample(range(len(node.state)), 2)
            temp[i] = 1 - temp[i]  # Flip the first bit
            temp[j] = 1 - temp[j]  # Flip the second bit

        new_node = Node(state=temp)
        val = heuristic_value_2(clause, new_node)

        if val > max_value:
            max_value = val
            max_node = new_node
    if max_node.state==node.state:
        return None
    return max_node
def gen_3(node, clause, num_neighbors=10):
    max_value = -1
    max_node = node

    # Generate `num_neighbors` random successors by flipping one, two, or three bits each time
    for _ in range(num_neighbors):
        temp = node.state.copy()
        
        # Randomly choose between flipping 1, 2, or 3 bits
        num_bits_to_flip = random.choice([1, 2, 3])

        if num_bits_to_flip == 1:
            i = random.randint(0, len(node.state) - 1)
            temp[i] = 1 - temp[i]  # Flip one bit

        elif num_bits_to_flip == 2:
            i, j = random.sample(range(len(node.state)), 2)
            temp[i] = 1 - temp[i]  # Flip the first bit
            temp[j] = 1 - temp[j]  # Flip the second bit

        elif num_bits_to_flip == 3:
            i, j, k = random.sample(range(len(node.state)), 3)
            temp[i] = 1 - temp[i]  # Flip the first bit
            temp[j] = 1 - temp[j]  # Flip the second bit
            temp[k] = 1 - temp[k]  # Flip the third bit

        new_node = Node(state=temp)
        val = heuristic_value_2(clause, new_node)

        if val > max_value:
            max_value = val
            max_node = new_node
    if max_node.state==node.state:
        return None
    return max_node

def calculate_penetrance(num_instances, k, m, n):
    solved_count = 0
    
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        is_solved = vgn(clauses, k,m,n)
        
        if is_solved:
            solved_count += 1
    
    penetrance = (solved_count / num_instances) * 100
    return penetrance
    
def hill_climb(clause,node,gen_func, k,m,n,max_iter=1000):
    for i in range(max_iter):
        if check(clause,node):
            print(f"clause is {clause}")
            print("Solution found")
            print(f"Solution is{node.state}")
            print(f"Steps required to reach solution {i}")
            return node
        temp_node=gen_func(node,clause)
        if(temp_node==None):
            print("Local minima reached")
            return node
        node=temp_node
def vgn(clause, k,m,n):
    #node = Node([random.choice([0, 1]) for _ in range(n)])
    node=Node([0]*n)
    node=hill_climb(clause,node,gen_1,k,m,n)
    if(check(clause,node)):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_1")
        return node
    node=hill_climb(clause,node,gen_2,k,m,n)
    if(check(clause,node)):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_2")
        return node
    node=hill_climb(clause,node,gen_3,k,m,n)
    if(check(clause,node)):
        print("Solution found")
        print(f"Solution is{node.state}")
        print(f"Node reached after gen_3")
        return node
clause=generate_k_sat_problem(3,6,4)
clause=[(1, 2, 3), (-1, -2, 3), (1, -2, -3)]
print(clause)
print(vgn(clause,3,3,3))
#print(calculate_penetrance(5,3,6,6))
