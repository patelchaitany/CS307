import random

def generate_k_sat_problem(k, m, n):
    clauses = []

    for _ in range(m):
        clause = set()  
        
        while len(clause) < k:
            var = random.randint(1, n) 
            is_negated = random.choice([True, False])  
            literal = -var if is_negated else var
            clause.add(literal)
        
    
        clauses.append(sorted(clause, key=abs))
    
    return clauses

# # Example usage
# k = 3   # Number of literals per clause
# m = 5   # Number of clauses
# n = 4   # Number of distinct variables

# k_sat_problem = generate_k_sat_problem(k, m, n)
# print("Generated k-SAT Problem:")
# for clause in k_sat_problem:
#     print(clause)
