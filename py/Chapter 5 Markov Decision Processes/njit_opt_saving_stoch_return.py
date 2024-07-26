#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#              DYNAMIC PROGRAMMING BY JOHN STACHURSKI AND THOMAS SARGENT                     #
#                                                                                            #
# This code is used for Chapter 5 Markov DECISION PROCESSES:                                 #
# Application: OPTIMAL SAVING WITH Stochastic RETURN                                         #
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
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import LinearOperator
import quantecon as qe



# --------------------------------------------------------------------------------------------#
#                        USE NAMEDTUPLE TO STORE MODEL PARAMETERS                             #
# --------------------------------------------------------------------------------------------#

Optimal_Saving_Stoch_Return = namedtuple("saving_stoch", 
                                         ("β",                          # discount rate
                                          "γ",                          # CRRA
                                          "H",                          # η space
                                          "η_size",                     # η space size
                                          "ϕ",                          # η distribution
                                          "W",                          # Wealth state space
                                          "w_size",                     # Wealth space size
                                          "ρ",                          # Tauchen
                                          "ν",                          # Tauchen
                                          "m",                          # Tauchen
                                          "y_size"                      # income cardinality
                                         ))



#---------------------------------------------------------------------------------------------#
#                       CREATE A FUNCTION TO INPUT MODEL PARAMETERS                           #
#---------------------------------------------------------------------------------------------#


def create_saving_stoch_return_model(β=0.98, 
                                     γ=2.5,
                                     η_min=0.75,
                                     η_max=1.25,
                                     η_size=2,
                                     w_min=0.01,
                                     w_max=20.0,
                                     w_size=100,
                                     ρ=0.9,
                                     ν=0.1,
                                     m=3,
                                     y_size=20):
    H = np.linspace(η_min, η_max, η_size)
    ϕ = np.ones(η_size) * (1/η_size)
    W = np.linspace(w_min, w_max, w_size)

    return Optimal_Saving_Stoch_Return(β=β, 
                                       γ=γ,
                                       H=H,
                                       η_size=η_size,
                                       ϕ=ϕ,
                                       W=W,
                                       w_size=w_size,
                                       ρ=ρ,
                                       ν=ν,
                                       m=m,
                                       y_size=y_size)

# saving_stoch = create_saving_stoch_return_model()
    

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
def Tauchen(saving_stoch):  
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
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


# saving_stoch = create_saving_stoch_return_model()
# Y,Q = Tauchen(saving_stoch)
# np.sum(Q, axis=1)


#---------------------------------------------------------------------------------------------#
#                                   Utility Function                                          #
#---------------------------------------------------------------------------------------------#

@njit
def u(c, saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    return ((c**(1-γ)))/(1-γ)



#--------------------------------------------------------------------------------------------#
#                                   BELLMAN EQUATION FOR V                                   #
#--------------------------------------------------------------------------------------------#
@njit
def B(v, saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    Y,Q = Tauchen(saving_stoch)
    
    W = np.reshape(W, (w_size, 1, 1, 1))
    Y = np.reshape(Y, (1, y_size, 1, 1))
    WP = np.reshape(W, (1, 1, 1, w_size))
    H = np.reshape(H, (1, 1, η_size, 1))
    
    v = np.reshape(v, (1, 1, 1, η_size, w_size, y_size))
    Q = np.reshape(Q, (1, y_size, 1, 1, 1, y_size))
    ϕ = np.reshape(ϕ, (1, 1, 1, η_size, 1))
    
    c = W+Y-(WP/H)
    EV_Q = np.sum(v * Q, axis=-1)
    EV = np.sum(EV_Q * ϕ, axis=3)
    
    return np.where(c>0, u(c, saving_stoch) + β * EV, -np.inf)

# saving_stoch = create_saving_stoch_return_model()
# v = np.zeros((100,20,2))
# B(v, saving_stoch)

#---------------------------------------------------------------------------------------------#
#                                     Greedy Policy                                           #
#---------------------------------------------------------------------------------------------#

@njit
def get_greedy(v, saving_stoch):
    return np.argmax(B(v, saving_stoch), axis=-1)



#---------------------------------------------------------------------------------------------#
#                                   BELLMAN OPERATOR                                          #
#---------------------------------------------------------------------------------------------#

@njit
def T(v, saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    new_B = B(v, saving_stoch)
    new_v = np.zeros((w_size, y_size, η_size))
    for i in range(w_size):
        for j in range(y_size):
            for k in range(η_size):
                new_v[i,j,k] = np.max(new_B[i,j,k, :])
    return new_v



# saving_stoch = create_saving_stoch_return_model()
# v = np.zeros((100,20,2))
# T(v, saving_stoch)

#---------------------------------------------------------------------------------------------#
#                                SUCCESSIVE APPROXIMATION                                     #
#---------------------------------------------------------------------------------------------#

@njit
def successive_approx (T,                                        # A callable operator
                       v_init,                                   # Initial condition
                       saving_stoch,                               # Model parameter
                       tol = 1e-6,                               # Error tolerance
                       max_iter = 10_000,                        # max iterations
                       print_step = 25                           # Print at multiples of print_step
                      ):
    v = v_init                                                   # set the initial condition
    error = tol + 1                                              # Initialize the error
    k = 0                                                        # initialize the iteration
    
    while error > tol and k < max_iter: 
        new_v = T(v,saving_stoch)                                  # update by applying operator T
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
#                                   POLICY OPERATOR                                          #
#--------------------------------------------------------------------------------------------#

# for loop to be compatible with numba
# The problem is in np.empty_like 
# If we initialize Σ = np.zeros((w_size, y_size))
@njit
def T_σ(v,σ,saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    Y,Q = Tauchen(saving_stoch)
    
    # Σ = np.empty_like(σ)
    Σ = np.zeros((w_size,y_size, η_size))
    
    for i in range(w_size):
        for j in range(y_size):
            for k in range(η_size):
                Σ[i,j,k] = W[σ[i,j,k]]
    W = np.reshape(W, (w_size, 1, 1))
    Y = np.reshape(Y, (1, y_size, 1))
    H = np.reshape(H, (1, 1, η_size))
    c = W + Y - (Σ/H)

    EV = np.zeros((w_size, y_size, η_size))
    for i in np.arange(w_size):
        for j in np.arange(y_size):
            for k in np.arange(η_size):
                EV[i,j,k] = np.sum(np.array([np.sum(np.array([v[σ[i,j,k],l,n] *  Q[j,l] for l in np.arange(y_size)]))*ϕ[n] for n in np.arange(η_size)]))

    return np.where(c>0, u(c, saving_stoch) + β * EV, -np.inf)



def T_σ_vec(v,σ,saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    Y,Q = Tauchen(saving_stoch)
    Σ = W[σ]
    W = np.reshape(W, (w_size, 1, 1))
    Y = np.reshape(Y, (1, y_size, 1))
    H = np.reshape(H, (1, 1, η_size))
    c = W + Y - (Σ/H)

    V = v[σ]
    Q = np.reshape(Q, (1,y_size, 1 ,y_size))
    ϕ = np.reshape(ϕ, (1, 1, 1, η_size, 1))
    
    EV_Q = np.sum(V * Q, axis=-2)
    EV = np.sum(EV_Q * ϕ, axis=-1)

    
    return np.where(c>0, u(c, saving_stoch) + β * EV, -np.inf)



#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#                                      ALGORITHMS                                             #
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------#
#                                VALUE FUNCTION ITERATION                                     #
#---------------------------------------------------------------------------------------------#

def value_function_iteration(saving_stoch):
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    v_init = np.zeros((w_size, y_size, η_size))
    v_star = successive_approx(T, v_init, saving_stoch)
    σ_star = get_greedy(v_star, saving_stoch)
    return v_star,  σ_star



#---------------------------------------------------------------------------------------------#
#                              OPTIMISTIC POLICY ITERATION                                    #
#---------------------------------------------------------------------------------------------#

# Currently OPI does not converge if we use T_σ (for loops indexing with numba)

# OPI converges if we use T_σ_vec (multi-dimensional indexing without numba)

def optimistic_policy_iteration(saving_stoch,
                                M=100,
                                tol=1e-6, 
                                max_iter=10_000,
                                print_step=25):
    
    β, γ, H, η_size, ϕ, W, w_size, ρ, ν, m, y_size = saving_stoch
    v = np.zeros((w_size, y_size, η_size))
    error = tol+1
    k = 0 

    while error > tol and k < max_iter:
        last_v = v
        σ = get_greedy(last_v,saving_stoch)
        for i in range(M):
            v = T_σ_vec(v, σ, saving_stoch)
        error = np.max(np.abs(last_v-v))
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.")
        k += 1
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
        v_star_opi = v
        σ_star_opi = get_greedy(v_star_opi, saving_stoch)
    else:     
        print("Warning: hit iteration bound.")
    return v_star_opi, σ_star_opi



#---------------------------------------------------------------------------------------------#
#                                         PLAYGROUND                                          #
#---------------------------------------------------------------------------------------------#

start_time = time.time()

#-------------ON YOUR MARKS---------SET----------------BANG!----------------------------------#


saving_stoch = create_saving_stoch_return_model()
optimistic_policy_iteration(saving_stoch)

#-------------------------------------------------------------------------------------------#
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")







