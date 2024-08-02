#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                DYNAMIC PROGRAMMING BY JOHN STARCHURSKI AND THOMAS SARGENT                  #
#                                                                                            #
# This code is used for Chapter 3 Markov Dynamics: Job-Search Model with Markov wages        #
# Written by Longye Tian 24/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------#
#                              IMPORT LIBRARIES AND PACKAGES                                 #
#--------------------------------------------------------------------------------------------#

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

#--------------------------------------------------------------------------------------------#
#                          CREATE NAMEDTUPLE TO STORE MODEL PARAMETERS                       #
#--------------------------------------------------------------------------------------------#

Job_Search_RS = namedtuple("job_search_rs", 
                               ("w_size",                      # wage grid size
                                "m",                           # number of std
                                "ρ",                           # wage persistence
                                "ν",                           # wage volatility coefficent
                                "β",                           # discount factor
                                "c",                           # unemployment compensation
                                "θ"                            # Risk sensitivity
                               ))      

def create_job_search_rs_model(w_size=200, m=3,ρ=0.9, ν=0.2, β=0.98, c=1.0, θ=5):
    A = np.array((0,1))
    return Job_Search_RS(w_size=w_size,m=m,ρ=ρ, ν=ν, β=β, c=c, θ=θ)



#--------------------------------------------------------------------------------------------#
#                                      NORMAL CDF                                            #
#--------------------------------------------------------------------------------------------#

@njit
def norm_cdf(x, mean=0, std=1):
    # Transform x to the standard normal
    z = (x - mean) / std
    
    # Use the Abramowitz & Stegun approximation for standard normal
    t = 1 / (1 + 0.2316419 * np.abs(z))
    d = 0.3989423 * np.exp(-z * z / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    return 1 - p if z > 0 else p


#--------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                      #
#--------------------------------------------------------------------------------------------#

@njit
def Tauchen(job_search_rs):  
    w_size,m,ρ,ν,β,α,c = job_search_rs             # Unpack model parameters
    σ_w = np.sqrt(ν**2/(1-ρ**2))                               # W's std
    W = np.linspace(-m*σ_w, m*σ_w, w_size)                     # State space by Tauchen
    s = (W[w_size-1]-W[0])/(w_size-1)                               # gap between two states
    P = np.zeros((w_size,w_size))                              # Initialize P
    for i in range(w_size):
        P[i,0] = norm_cdf(W[0]-ρ*W[i]+s/2, std=σ_w)            # j=1
        P[i,w_size-1] = 1 - norm_cdf(W[w_size-1]-ρ*W[i]-s/2, std=σ_w)    # j=n
        for j in range(1,w_size-1):
            P[i,j] = norm_cdf(W[j]-ρ*W[i]+s/2, std=σ_w)-norm_cdf(W[j]-ρ*W[i]-s/2, std=σ_w)
    W = np.exp(W)
    return W,P


# job_search_rs = create_job_search_rs_model()
# W,P = Tauchen(job_search_rs)

# Check the row sum
# np.sum(P, axis=1)


#--------------------------------------------------------------------------------------------#
#                   VALUE AGGREGATOR IN RISK SENSITIVE JOB SEARCH MODEL                      #
#--------------------------------------------------------------------------------------------#

def T (v, job_search_rs):
    w_size ,m,ρ,ν,β,c, θ = job_search_rs             # Unpack model parameters
    W,P = Tauchen(job_search_rs)
    
    v = np.exp(θ * v)                                        
    h = c + β/θ * np.log(np.matmul(P, v))
    e = W/(1-β)                                                # Stopping value

    return np.maximum(e,h)


#--------------------------------------------------------------------------------------------#
#                                   v-greedy policy                                          #
#--------------------------------------------------------------------------------------------#

def get_greedy (v, job_search_rs): 
    w_size,m,ρ,ν,β,c,θ = job_search_rs                              # Unpack model parameters
    W,P = Tauchen(job_search_rs)                               # Get W and P
    
    v = np.exp(θ * v)                                        
    h = c + β/θ * np.log(np.matmul(P, v))
    e = W/(1-β)                                                 # Stopping value
    σ= np.where(e>=h, 1, 0)                                      
    return σ


#--------------------------------------------------------------------------------------------#
#                            SUCCESSIVE APPROXIMATION ALGORITHM                              #
#--------------------------------------------------------------------------------------------#

def successive_approx (T,                                         # A callable operator
                       v_init,                                    # Initial condition
                       job_search_rs,                             # Model parameter
                       tol = 1e-6,                                # Error tolerance
                       max_iter = 10_000,                         # max iterations
                       print_step = 25                            # Print at multiples of print_step
                      ):
    v = v_init                                                    # set the initial condition
    error = tol + 1                                               # Initialize the error
    k = 0                                                         # initialize the iteration
    
    while (error > tol) and (k <= max_iter): 
        v_new = T(v,job_search_rs)                                # update by applying operator T
        error = np.max(np.abs(v_new-v))                           # update the error
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.") 
        v = v_new                                                 # update x
        k += 1                                                    # update the steps
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return v                                                     


#--------------------------------------------------------------------------------------------#
#                                 VALUE FUNCTION ITERATION                                   #
#--------------------------------------------------------------------------------------------#

def value_function_iteration (job_search_rs):
    w_size,m,ρ,ν,β,c, θ = job_search_rs                           # Unpack model parameters
    W,P = Tauchen(job_search_rs)                                  # Get W and P
    
    v_init = W/(1-β)                                              # initialize Value function
    v_star = successive_approx(T, v_init, job_search_rs)    
    σ_star = get_greedy(v_star, job_search_rs)                    # σ* is v*-greedy
    return v_star, σ_star



#--------------------------------------------------------------------------------------------#
#                   PLOT CONTINUATION, STOPPING AND VALUE FUNCTIONS                          #
#--------------------------------------------------------------------------------------------#

def plot_value_stopping_continue (job_search_rs):
    w_size,m,ρ,ν,β,c, θ = job_search_rs                           # Unpack model parameters
    W,P= Tauchen(job_search_rs)                                   # Get W and P
    v_star, σ_star = value_function_iteration(job_search_rs) 
    v_star = np.exp(θ*v_star)  
    h = c + β/θ * np.log(np.matmul(P,v_star))     
    e = W/(1-β)                                                  # Stopping value
    plt.plot(W, h, label='Continuation Value')
    plt.plot(W, e, label='Stopping Value')
    plt.plot(W, v_star, label='Value function')
    plt.legend()
    plt.show()
    
# job_search_markov = create_job_search_markov_model()
# plot_value_stopping_continue (job_search_markov)
