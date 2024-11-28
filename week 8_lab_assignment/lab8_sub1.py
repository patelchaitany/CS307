import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class poisson_class:
    def __init__(self, rental_rate,return_rate,max_value = 20):
        self.apha = rental_rate
        self.beta = return_rate
        self.max_value = max_value+1

        rental_prob= self.poisson_probability(self.apha)
        self.rental_prob = rental_prob/rental_prob.sum()
        return_prob= self.poisson_probability(self.beta)
        self.return_prob = return_prob/return_prob.sum()
        
    def get_rental(self, num):
        return self.rental_prob[num]
    
    def get_return(self, num):
        return self.return_prob[num]

    def poisson_probability(self,lambda_value):

        k = np.arange(0,self.max_value)

        return (lambda_value**k * np.exp(-lambda_value)) / factorial(k)


def make_action(state,action):
    max_value = 20
    return [max(0, min(state[0] + action,max_value)), max(0, min(state[1] - action,max_value))]


loc_a = poisson_class(3,4)
loc_b = poisson_class(3,2)

value = np.zeros((20+1, 20+1))
policy = value.copy().astype(int)

gam = 0.9

def expected_return(state,action):
    global value
    r = 0

    new_state = make_action(state,action)
    
    if action <= 0:
        r = r + -2*abs(action)
    else:
        r = r + -2*abs(action-1)
    
    if new_state[0] > 10:
        r = r + (-10)
    if  new_state[1] > 10:
        r = r + (-10)

    for rent_a in range(0,21):
        for rent_b in range(0,21):
            for return_a in range(0,21):
                for return_b in range(0,21):
                    prob = loc_a.get_rental(rent_a) * loc_b.get_rental(rent_b) * loc_a.get_return(return_a) * loc_b.get_return(return_b)

                    valid_requests_A = min(new_state[0], rent_a)
                    valid_requests_B = min(new_state[1], rent_b)

                    reward = (valid_requests_A + valid_requests_B)*(10)

                    new_s = [0,0]
                    new_s[0] = max(min(new_state[0] - valid_requests_A + return_a, 20),0)
                    new_s[1] = max(min(new_state[1] - valid_requests_B + return_b, 20),0)

                    r = r + prob*(reward + gam*value[new_s[0]][new_s[1]])

    return r

error = 50
def policy_evaluation():
    global value
    global error
    error_1 = error/10

    error = error/10
    while True:
        
        delta = 0 
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):

                old_val = value[i][j]
            
                value[i][j] = expected_return([i,j],policy[i][j])
                
                delta = max(delta ,abs(old_val - value[i][j]))
                print('.', end = '')
                sys.stdout.flush()

        print(delta)
        sys.stdout.flush()
        if delta < error_1:
            break
                

def policy_improvement():
    
    global policy
    
    policy_stable = True
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            old_action = policy[i][j]
            
            max_act_val = None
            max_act = None
            
            t12 = min(i,5)      
            t21 = -min(j,5)                
            for act in range(t21,t12+1):
                σ = expected_return([i,j], act)
                if max_act_val == None:
                    max_act_val = σ
                    max_act = act
                elif max_act_val < σ:
                    max_act_val = σ
                    max_act = act
                
            policy[i][j] = max_act
            
            if old_action!= policy[i][j]:
                policy_stable = False
    
    return policy_stable

save_policy_counter = 0
save_value_counter = 0
def save_policy():
    
    global save_policy_counter
    save_policy_counter += 1
    ax = sns.heatmap(policy, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('policy'+str(save_policy_counter)+'.svg')
    plt.close()
    
def save_value():
    global save_value_counter
    save_value_counter += 1
    ax = sns.heatmap(value, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('value'+ str(save_value_counter)+'.svg')
    plt.close()

if __name__ == "__main__":
    while(1):
        policy_evaluation()
        ρ = policy_improvement()
        save_value()
        save_policy()
        if ρ == True:
            break
