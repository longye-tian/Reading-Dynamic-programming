#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#              DYNAMIC PROGRAMMING BY JOHN STACHURSKI AND THOMAS SARGENT                     #
#                                                                                            #
# This code is used for Chapter 5 Markov DECISION PROCESSES:                                 #
# Application: OPTIMAL SAVING WITH LABOR INCOME                                              #
# Improved computation efficiency using numba.njit                                           #
# Written by Longye Tian 02/07/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                               IMPORT LIBRARIES AND PACKAGES                                #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from numba import njit                                       
import time


# --------------------------------------------------------------------------------------------#
#                        USE NAMEDTUPLE TO STORE MODEL PARAMETERS                             #
# --------------------------------------------------------------------------------------------#

Optimal_Saving_MDP = namedtuple("saving_mdp", 
                                ("R",                          # gross interest rate
                                 "β",                          # discount rate
                                 "γ",                          # CRRA
                                 "W",                          # Wealth state space
                                 "w_size",
                                 "ρ",                          # Tauchen
                                 "ν",                          # Tauchen
                                 "m",                          # Tauchen
                                 "y_size"                      # income cardinality
                                ))


#---------------------------------------------------------------------------------------------#
#                       CREATE A FUNCTION TO INPUT MODEL PARAMETERS                           #
#---------------------------------------------------------------------------------------------#

def create_saving_mdp_model(R=1.01, 
                            β=0.98, 
                            γ=2.5, 
                            w_min=0.01, 
                            w_max=20.0, 
                            w_size=200, 
                            ρ=0.9, 
                            ν=0.1, 
                            m=3,
                            y_size=5):
    W = np.linspace(w_min, w_max, w_size)
    return Optimal_Saving_MDP(R=R, 
                              β=β, 
                              γ=γ, 
                              W=W,
                              w_size=w_size,
                              ρ=ρ, 
                              ν=ν, 
                              m=m,
                              y_size=y_size)

# saving_mdp = create_saving_mdp_model()


#---------------------------------------------------------------------------------------------#
#                                      NORMAL CDF                                             #
#---------------------------------------------------------------------------------------------#

@njit
def norm_cdf(x, mean=0, std=1):
    # Transform x to the standard normal
    z = (x - mean) / std
    
    # Use the Abramowitz & Stegun approximation for standard normal
    t = 1 / (1 + 0.2316419 * np.abs(z))
    d = 0.3989423 * np.exp(-z * z / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    return 1 - p if z > 0 else p



#---------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                       #
#---------------------------------------------------------------------------------------------#

@njit
def Tauchen(saving_mdp):  
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    σ_y = np.sqrt(ν**2/(1-ρ**2))                               # W's std
    Y = np.linspace(-m*σ_y, m*σ_y, y_size)                     # State space by Tauchen
    s = (Y[y_size-1]-Y[0])/(y_size-1)                          # gap between two states
    Q = np.zeros((y_size,y_size))                              # Initialize Q
    for i in range(y_size):
        Q[i,0] = norm_cdf(Y[0]-ρ*Y[i]+s/2, std=σ_y)            
        Q[i,y_size-1] = 1 - norm_cdf(Y[y_size-1]-ρ*Y[i]-s/2, std=σ_y)   
        for j in range(1,y_size-1):
            Q[i,j] = norm_cdf(Y[j]-ρ*Y[i]+s/2, std=σ_y)-norm_cdf(Y[j]-ρ*Y[i]-s/2, std=σ_y)
    Y = np.exp(Y)
    return Y,Q


# saving_mdp = create_saving_mdp_model()
# Y,Q = Tauchen(saving_mdp)

#---------------------------------------------------------------------------------------------#
#                                   Utility Function                                          #
#---------------------------------------------------------------------------------------------#

@njit
def u(c, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    return (c**(1-γ))/(1-γ)

#--------------------------------------------------------------------------------------------#
#                                   BELLMAN EQUATION FOR V                                   #
#--------------------------------------------------------------------------------------------#
@njit
def B(v, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    
    W = np.reshape(W, (w_size, 1, 1))
    Y = np.reshape(Y, (1, y_size, 1))
    WP = np.reshape(W, (1, 1, w_size))
    
    v = np.reshape(v, (1, 1, w_size, y_size))
    Q = np.reshape(Q, (1, y_size, 1, y_size))
    
    c = W+Y-(WP/R)
    EV = np.sum(v * Q, axis=-1)
    
    return np.where(c>0, u(c, saving_mdp) + β * EV, -np.inf)

# saving_mdp = create_saving_mdp_model()
# v = np.zeros((200,5))
# B(v, saving_mdp)

#---------------------------------------------------------------------------------------------#
#                                     Greedy Policy                                           #
#---------------------------------------------------------------------------------------------#

@njit
def get_greedy(v, saving_mdp):
    return np.argmax(B(v, saving_mdp), axis=-1)



#---------------------------------------------------------------------------------------------#
#                                   BELLMAN OPERATOR                                          #
#---------------------------------------------------------------------------------------------#

@njit
def T(v, saving_mdp):
    new_B = B(v, saving_mdp)
    w_size = new_B.shape[2]
    new_v = np.empty(new_B.shape[:2])
    for i in range(new_B.shape[0]):
        for j in range(new_B.shape[1]):
            new_v[i, j] = np.max(new_B[i, j, :])
    return new_v




# saving_mdp = create_saving_mdp_model()
# v = np.zeros((200,5))
# T(v, saving_mdp)

    


#---------------------------------------------------------------------------------------------#
#                                SUCCESSIVE APPROXIMATION                                     #
#---------------------------------------------------------------------------------------------#

@njit
def successive_approx (T,                                        # A callable operator
                       v_init,                                   # Initial condition
                       saving_mdp,                               # Model parameter
                       tol = 1e-6,                               # Error tolerance
                       max_iter = 10_000,                        # max iterations
                       print_step = 25                           # Print at multiples of print_step
                      ):
    v = v_init                                                   # set the initial condition
    error = tol + 1                                              # Initialize the error
    k = 0                                                        # initialize the iteration
    
    while error > tol and k < max_iter: 
        new_v = T(v,saving_mdp)                                  # update by applying operator T
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

    

#---------------------------------------------------------------------------------------------#
#                                VALUE FUNCTION ITERATION                                     #
#---------------------------------------------------------------------------------------------#

def value_function_iteration(saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    v_init = np.zeros((w_size, y_size), dtype=np.float64)
    v_star = successive_approx(T, v_init, saving_mdp)
    σ_star = get_greedy(v_star, saving_mdp)
    return v_star,  σ_star
#--------------------------------------------------------------------------------------------#
#                                   POLICY OPERATOR                                          #
#--------------------------------------------------------------------------------------------#

@njit
def T_σ(v,σ,saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    new_v = np.empty_like(v)
    Y,Q = Tauchen(saving_mdp)
    for i in np.arange(len(W)):
        for j in np.arange(y_size):
            new_v[i,j]= B(i,j,σ[i,j],v,saving_mdp)
    return new_v

#---------------------------------------------------------------------------------------------#
#                                  POLICY EVALUATION                                          #
#---------------------------------------------------------------------------------------------#

@njit
def get_value(σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp


#---------------------------------------------------------------------------------------------#
#                                         PLAYGROUND                                          #
#---------------------------------------------------------------------------------------------#

start_time = time.time()

#-------------ON YOUR MARKS---------SET----------------BANG!----------------------------------#


saving_mdp = create_saving_mdp_model()
v_star,  σ_star = value_function_iteration(saving_mdp)



#-------------------------------------------------------------------------------------------#
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
