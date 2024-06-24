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

import numpy as np
from collections import namedtuple
from scipy.stats import norm
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------#
#                          CREATE NAMEDTUPLE TO STORE MODEL PARAMETERS                       #
#--------------------------------------------------------------------------------------------#

Job_Search_Markov = namedtuple("job_search_markov", 
                               ("n",                           # wage grid size
                                "m",                           # number of std
                                "ρ",                           # wage persistence
                                "ν",                           # wage volatility coefficent
                                "β",                           # discount factor
                                "c"                            # unemployment compensation
                               ))      

def create_job_search_markov_model(n=200, m=3,ρ=0.9, ν=0.2, β=0.98, c=1.0):
    return Job_Search_Markov(n=n,m=m,ρ=ρ, ν=ν, β=β, c=c)



#--------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                      #
#--------------------------------------------------------------------------------------------#

def Tauchen(job_search_markov):  
    n,m,ρ,ν,β,c = job_search_markov                              # Unpack model parameters
    σ_w = np.sqrt(ν**2/(1-ρ**2))                               # W's std
    W = np.linspace(-m*σ_w, m*σ_w, n)                          # State space by Tauchen
    s = (W[n-1]-W[0])/(n-1)                                    # gap between two states
    P = np.zeros((n,n))                                        # Initialize P
    for i in range(n):
        P[i,0] = norm.cdf(W[0]-ρ*W[i]+s/2, scale=σ_w)          # j=1
        P[i,n-1] = 1 - norm.cdf(W[n-1]-ρ*W[i]-s/2, scale=σ_w)  # j=n
        for j in range(1,n-1):
            P[i,j] = norm.cdf(W[j]-ρ*W[i]+s/2, scale=σ_w)-norm.cdf(W[j]-ρ*W[i]-s/2, scale=σ_w)
    return W,P
    


# job_search_markov = create_job_search_markov_model()
# W,P = Tauchen(job_search_markov)

# Check the row sum
# for i in range(n):
#    if np.sum(P[i])!=1:
#        print(i, np.sum(P[i]))


#--------------------------------------------------------------------------------------------#
#                    BELLMAN OPERATOR IN JOB SEARCH MDOEL WITH MARKOV WAGE                   #
#--------------------------------------------------------------------------------------------#

def T (v, job_search_markov):
    n,m,ρ,ν,β,c = job_search_markov                              # Unpack model parameters
    W,P = Tauchen(job_search_markov)                             # Get W and P
    h = np.asarray(c + β * np.matmul(P,v))                       # Continuation value
    e = W/(1-β)                                                  # Stopping value
    return np.maximum(e,h)


#--------------------------------------------------------------------------------------------#
#                                   v-greedy policy                                          #
#--------------------------------------------------------------------------------------------#

def get_greedy (v, job_search_markov): 
    n,m,ρ,ν,β,c = job_search_markov                              # Unpack model parameters
    W,P = Tauchen(job_search_markov)                             # Get W and P
    h = np.asarray(c + β * np.matmul(P,v))                       # Continuation value
    e = W/(1-β)                                                  # Stopping value
    σ= np.where(e>=h, 1, 0)                                      
    return σ


#--------------------------------------------------------------------------------------------#
#                            SUCCESSIVE APPROXIMATION ALGORITHM                              #
#--------------------------------------------------------------------------------------------#

def successive_approx (T,                                         # A callable operator
                       v_0,                                       # Initial condition
                       job_search_markov,                         # Model parameter
                       tol = 1e-6,                                # Error tolerance
                       max_iter = 10_000,                         # max iterations
                       print_step = 25                            # Print at multiples of print_step
                      ):
    v = v_0                                                       # set the initial condition
    error = tol + 1                                               # Initialize the error
    k = 1                                                         # initialize the iteration
    
    while (error > tol) and (k <= max_iter): 
        v_new = T(v,job_search_markov)                            # update by applying operator T
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

def value_function_iteration (job_search_markov):
    n,m,ρ,ν,β,c = job_search_markov                              # Unpack model parameters
    v_init = np.zeros(n)                                         # initialize Value function
    v_star = successive_approx(T, v_init, job_search_markov)    
    σ_star = get_greedy(v_star, job_search_markov)                # σ* is v*-greedy
    return v_star, σ_star



#--------------------------------------------------------------------------------------------#
#                   PLOT CONTINUATION, STOPPING AND VALUE FUNCTIONS                          #
#--------------------------------------------------------------------------------------------#

def plot_value_stopping_continue (job_search_markov):
    n,m,ρ,ν,β,c = job_search_markov                              # Unpack model parameters
    W,P= Tauchen(job_search_markov)                             # Get W and P
    v_star, σ_star = value_function_iteration(job_search_markov) 
    h = np.asarray(c + β * np.matmul(P,v_star))                  # Continuation value function
    e = W/(1-β)                                                  # Stopping value
    plt.plot(W, h, label='Continuation Value')
    plt.plot(W, e, label='Stopping Value')
    plt.plot(W, v_star, label='Value function')
    plt.legend()
    plt.show()
    
# job_search_markov = create_job_search_markov_model()
# plot_value_stopping_continue (job_search_markov)