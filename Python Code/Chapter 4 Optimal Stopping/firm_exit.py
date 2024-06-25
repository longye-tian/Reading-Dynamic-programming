#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                Dynamic Programming by John Stachurski and Tom Sargent                      #
#                                                                                            #
# This code is used for Chapter 4 OPTIMAL STOPPING: FIRM EXIT MODEL                          #
# Written by Longye Tian 25/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                              IMPORT LIBRARIES AND PACKAGES                                 #
#--------------------------------------------------------------------------------------------#

import numpy as np
from collections import namedtuple
from scipy.stats import norm
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------#
#                          CREATE NAMEDTUPLE TO STORE MODEL PARAMETERS                       #
#--------------------------------------------------------------------------------------------#

Firm_Exit = namedtuple("firm_exit",("n",                        # Productivity grid size
                                    "m",                        # range 
                                    "ρ",                        # Persistence
                                    "μ",                        # Mean
                                    "ν",                        # Volatility
                                    "β",                        # Discount rate
                                    "s"                         # Scrap value
                                   ))

def create_firm_exit_model(n=200, m=3, ρ=0.95,μ=0.1,ν=0.1,β=0.98,s=0.5):
    return Firm_Exit(n=n,m=m,ρ=ρ,μ=μ,ν=ν,β=β,s=s)

# firm_exit = create_firm_exit_model()


#--------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                      #
#--------------------------------------------------------------------------------------------#

def Tauchen(firm_exit):  
    n,m,ρ,μ,ν,β,s = firm_exit                                  # Unpack model parameters
    σ_z = np.sqrt(ν**2/(1-ρ**2))                               # W's std
    Z = np.linspace(-m*σ_z+μ, m*σ_z+μ, n)                      # State space by Tauchen
    d = (Z[n-1]-Z[0])/(n-1)                                    # gap between two states
    Q = np.zeros((n,n))                                        # Initialize P
    for i in range(n):
        Q[i,0] = norm.cdf(Z[0]-ρ*Z[i]+d/2, scale=σ_z)          # j=1
        Q[i,n-1] = 1 - norm.cdf(Z[n-1]-ρ*Z[i]-d/2, scale=σ_z)  # j=n
        for j in range(1,n-1):
            Q[i,j] = norm.cdf(Z[j]-ρ*Z[i]+d/2, scale=σ_z)-norm.cdf(Z[j]-ρ*Z[i]-d/2, scale=σ_z)
    return Z,Q
    


# firm_exit = create_firm_exit_model()
# Z,P = Tauchen(firm_exit)

# Check the row sum
# for i in range(n):
#    if np.sum(P[i])!=1:
#        print(i, np.sum(P[i]))


#--------------------------------------------------------------------------------------------#
#                          BELLMAN OPERATOR FOR FIRM EXIT MODEL                              #
#--------------------------------------------------------------------------------------------#

def T(v, firm_exit):
    n,m,ρ,μ,ν,β,s = firm_exit                                 # Unpack model parameters
    Z,Q = Tauchen(firm_exit)                                  # Get Z,P
    S = s * np.ones(n)                                        # Stopping value function
    h = np.asarray(Z + β * np.matmul(Q,v))                    # Continuation value function
    return np.maximum(S,h)


#--------------------------------------------------------------------------------------------#
#                                   v-greedy policy                                          #
#--------------------------------------------------------------------------------------------#

def get_greedy(v, firm_exit):
    n,m,ρ,μ,ν,β,s = firm_exit                                 # Unpack model parameters
    Z,Q = Tauchen(firm_exit)                                  # Get Z,P
    S = s * np.ones(n)                                        # Stopping value function
    h = np.asarray(Z + β * np.matmul(Q,v))                    # Continuation value function
    σ = np.where(S>=h, 1, 0)                                  # v-greedy policy
    return σ


#--------------------------------------------------------------------------------------------#
#                            SUCCESSIVE APPROXIMATION ALGORITHM                              #
#--------------------------------------------------------------------------------------------#

def successive_approx (T,                                     # A callable operator
                       v_0,                                   # Initial condition
                       firm_exit,                             # Model parameter
                       tol = 1e-6,                            # Error tolerance
                       max_iter = 10_000,                     # max iterations
                       print_step = 25                        # Print at multiples of print_step
                      ):
    v = v_0                                                   # set the initial condition
    error = tol + 1                                           # Initialize the error
    k = 1                                                     # initialize the iteration
    
    while (error > tol) and (k <= max_iter): 
        v_new = T(v,firm_exit)                                # update by applying operator T
        error = np.max(np.abs(v_new-v))                       # update the error
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.") 
        v = v_new                                             # update x
        k += 1                                                # update the steps
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return v    


#--------------------------------------------------------------------------------------------#
#                                 VALUE FUNCTION ITERATION                                   #
#--------------------------------------------------------------------------------------------#

def value_function_iteration (firm_exit):
    n,m,ρ,μ,ν,β,s = firm_exit                                 # Unpack model parameters
    v_init = np.zeros(n)                                      # initialize Value function
    v_star = successive_approx(T, v_init, firm_exit)    
    σ_star = get_greedy(v_star, firm_exit)                    # σ* is v*-greedy
    return v_star, σ_star


#--------------------------------------------------------------------------------------------#
#                   PLOT CONTINUATION, STOPPING AND VALUE FUNCTIONS                          #
#--------------------------------------------------------------------------------------------#

def plot_value_stopping_continue (firm_exit):
    n,m,ρ,μ,ν,β,s = firm_exit                                 # Unpack model parameters
    Z,Q = Tauchen(firm_exit)                                  # Get Z,P
    v_star, σ_star = value_function_iteration(firm_exit) 
    S = s * np.ones(n)                                        # Stopping value function
    h = np.asarray(Z + β * np.matmul(Q,v_star))               # Continuation value function
    plt.plot(Z, h, label='Continuation Value')
    plt.plot(Z, S, label='Stopping Value')
    plt.plot(Z, v_star, label='Value function')
    plt.legend()
    plt.show()
    
firm_exit = create_firm_exit_model()
plot_value_stopping_continue (firm_exit)