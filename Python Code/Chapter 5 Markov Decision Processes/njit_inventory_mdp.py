#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#               DYNAMIC PROGRAMMING BY JOHN STACHURSKI AND THOMAS SARGENT                    #
#                                                                                            #
# This code is used for Chapter 5 Markov DECISION PROCESSES:                                 #
# Application: OPTIMAL INVENTORY                                                             #
# Improved computation efficiency using numba.njit                                           #
# Written by Longye Tian 28/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                               IMPORT LIBRARIES AND PACKAGES                                #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple
from scipy.stats import geom
import matplotlib.pyplot as plt
from numba import njit                                       
import time


#--------------------------------------------------------------------------------------------#
#                        CREATE A NAMEDTUPLE TO STORE MODEL PARAMETERS                       #
#--------------------------------------------------------------------------------------------#


Inventory_MDP = namedtuple("inventory_mdp", ("β",                  # DISCOUNT FACTOR
                                             "K",                  # STORAGE CAPACITY
                                             "x_vals",             # STORAGE STATE SPACE
                                             "c",                  # UNIT COST
                                             "κ",                  # FIXED COST
                                             "p",                  # DISTRIBUTION PARAMETER
                                             "d_max",              # MAXIMUM d
                                             "D",                  # Demand space
                                             "ts_length"           # time length
                                            ))

#--------------------------------------------------------------------------------------------#
#                       CREATE A FUNCTION TO INPUT MODEL PARAMETERS                          #
#--------------------------------------------------------------------------------------------#

def create_inventory_mdp_model(β=0.98, K=40, c=0.2, κ=2, p=0.6, d_max=100, ts_length=400):
    x_vals = np.arange(K+1)
    D = np.arange(1,d_max+1)
    return Inventory_MDP(β=β, K=K, x_vals=x_vals, c=c, κ=κ, p=p, d_max=d_max, D=D, ts_length=ts_length)


#--------------------------------------------------------------------------------------------#
#                                 CREATE THE DEMAND PMF                                      #
#--------------------------------------------------------------------------------------------#

@njit
def demand_pmf(d,p):
    return (1-p)**(d-1)*p

#--------------------------------------------------------------------------------------------#
#                      BELLMAN EQUATION IN OPTIMAL INVENTORY MODEL                           #
#--------------------------------------------------------------------------------------------#

@njit
def B (x,a,v, inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp    # UNPACK PARAMETERS
    pmf = demand_pmf(D,p)                                         # pmf
    current_revenue = np.sum(np.minimum(x,D) * pmf)      
    current_profit = current_revenue - c*a - κ*(a>0)              # Profit
    next_value = np.sum(v[np.maximum(x-D,0)+a] * pmf)
    return current_profit + β*next_value

#--------------------------------------------------------------------------------------------#
#                                      BELLMAN OPERATOR                                      #
#--------------------------------------------------------------------------------------------#

@njit
def T(v, inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp    # UNPACK PARAMETERS
    new_v = np.empty_like(v)                                      # initialize the new value
    for x in x_vals:
        Γ_x = np.arange(K-x+1)
        new_v[x] = np.max(np.array([B(x,a,v,inventory_mdp) for a in Γ_x]))
    return new_v

#--------------------------------------------------------------------------------------------#
#                                      Greedy Policy                                         #
#--------------------------------------------------------------------------------------------#

@njit
def get_greedy(v, inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp    # UNPACK PARAMETERS
    σ = np.empty_like(x_vals)
    for x in x_vals:
        Γ_x = np.arange(K-x+1)
        σ[x] = np.argmax(np.array([B(x,a,v,inventory_mdp) for a in Γ_x])) 
    return σ

#--------------------------------------------------------------------------------------------#
#                                SUCCESSIVE APPROXIMATION                                    #
#--------------------------------------------------------------------------------------------#

@njit
def successive_approx (T,                                        # A callable operator
                       v_init,                                   # Initial condition
                       inventory_mdp,                            # Model parameter
                       tol = 1e-6,                               # Error tolerance
                       max_iter = 10_000,                        # max iterations
                       print_step = 25                           # Print at multiples of print_step
                      ):
    v = v_init                                                   # set the initial condition
    error = tol + 1                                              # Initialize the error
    k = 0                                                        # initialize the iteration
    
    while error > tol and k < max_iter: 
        new_v = T(v,inventory_mdp)                               # update by applying operator T
        error = np.max(np.abs(new_v-v))                          # update the error
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.") 
        v = new_v                                                # update x
        k += 1                                                   # update the steps
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return v   

#--------------------------------------------------------------------------------------------#
#                                VALUE FUNCTION ITERATION                                    #
#--------------------------------------------------------------------------------------------#

def value_function_iteration(inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp  
    v_init = np.zeros_like(x_vals, dtype=np.float64)
    v_star = successive_approx(T, v_init, inventory_mdp)
    σ_star = get_greedy(v_star, inventory_mdp)
    return v_star,  σ_star

#--------------------------------------------------------------------------------------------#
#                       PLOT VALUE FUNCTION AND OPTIMAL CHOICE AT EACH STATE                 #
#--------------------------------------------------------------------------------------------#

def plot_value_and_policy(inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp      
    v_star, σ_star = value_function_iteration(inventory_mdp)
    fig,axes = plt.subplots(2,1, figsize=(10,8))
    fig.suptitle('Value Function and Optimal Policy in Optimal Inventory MDP')
    
    axes[0].plot(x_vals, v_star, label='Value Function')
    axes[0].set_xlabel('Inventory Stock')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[1].step(x_vals, σ_star, label='Optimal Policy')
    axes[1].set_xlabel('Inventory Stock')
    axes[1].set_ylabel('Amount of Purchase')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
#--------------------------------------------------------------------------------------------#
#                               PLOT OPTIMAL INVENTORY DYNAMICS                              #
#--------------------------------------------------------------------------------------------#


def optimal_inventory_dynamics_mdp(inventory_mdp):
    β, K, x_vals, c, κ, p, d_max, D, ts_length = inventory_mdp       
    v_star, σ_star = value_function_iteration(inventory_mdp)
    X = np.zeros(ts_length)                                        
    T = np.arange(0,ts_length)
    d = np.random.geometric(p, size=ts_length) - 1
    for i in np.arange(1,ts_length):
        X[i] = np.maximum(X[i-1]-d[i-1],0)+ σ_star[int(X[i-1])] 
    fig, ax = plt.subplots()
    fig.suptitle('Optimal Inventory Dynamics')
    ax.plot(T,X, label='$X_t$')
    ax.legend()
    plt.show()
    
#--------------------------------------------------------------------------------------------#
#                                         PLAYGROUND                                         #
#--------------------------------------------------------------------------------------------#

start_time = time.time()

#------------ON YOUR MARKS---------SET----------------BANG!----------------------------------#


inventory_mdp = create_inventory_mdp_model()
plot_value_and_policy(inventory_mdp)
optimal_inventory_dynamics_mdp(inventory_mdp)



#-------------------------------------------------------------------------------------------#
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")