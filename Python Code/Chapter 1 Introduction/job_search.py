#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                Dynamic Programming by John Stachurski and Tom Sargent                      #
#                                                                                            #
# This code is used for Chapter 1 Introduction: Job Search model                             #
# Written by Longye Tian 22/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------#
#                                   Import Libraries and packages                            #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple


#--------------------------------------------------------------------------------------------#
#                      Create a namedtuple to store model parameters                         #
#--------------------------------------------------------------------------------------------#

Model = namedtuple("Model", ("c",                      # unemployment compensation
                             "w_vals",                 # state space (wage)
                             "n",                      # cardinality of state space
                             "β",                      # discount factor
                             "ts_length"               # number of periods
                            ))



#--------------------------------------------------------------------------------------------#
#                       Create a model with parameter values                                 #
#--------------------------------------------------------------------------------------------#

def create_job_search_model(n=50,                      # wage grid size
                            w_min=11,                  # lowest wage
                            w_max=60,                  # highest wage
                            c=10,                      # unemployment compensation
                            β=0.96,                    # discount factor
                            ts_length=10               # number of periods
                           ):
    w_vals = np.linspace(w_min,w_max, n)               # create the wage space
    return Model(c=c,                                  # return the namedtuple 
                 w_vals=w_vals, 
                 n=n, 
                 β=β, 
                 ts_length=ts_length) 


#--------------------------------------------------------------------------------------------#
#          Create a function to obtain continuation value and reservation wages              #
#--------------------------------------------------------------------------------------------#

def reservation_wage(model):
    c, w_vals, n, β, ts_length = model                 # Unpack the model parameters
    H = np.zeros(ts_length+1)                          # Initialize continuation value
    R = np.zeros(ts_length+1)                          # Initialize reservation wage 
    S = np.zeros((ts_length+1,n))                      # Initialize the maximum values
    H[ts_length] = c                                   # Last continuation value is c
    R[ts_length] = c                                   # Last rw is c
    S[ts_length,:] = np.maximum(c, w_vals)                     
    for t in range(1, ts_length+1):                    # Assume uniform distribution
        H[ts_length-t] = c + β * np.mean(S[ts_length-t+1,:]) 
        df = np.geomspace(1, β**t, t+1)                # denominator
        dfs = np.sum(df)                               # denominator for the rw
        R[ts_length-t] = H[ts_length+1-t]/dfs          # RW
        S[ts_length-t,:] = np.maximum(dfs * w_vals, H[ts_length-t])   
    return R

# model = create_job_search_model()
# reservation_wage(model)


#--------------------------------------------------------------------------------------------#
#                             Successive Approximation Algorithm                             #
#--------------------------------------------------------------------------------------------#

def successive_approx (T,                               # A callable operator
                       x_0,                             # Initial condition
                       model,                           # Model parameter
                       tol = 1e-6,                      # Error tolerance
                       max_iter = 10_000,               # max iterations
                       print_step = 25                  # Print at multiples of print_step
                      ):
    x = x_0                                             # set the initial condition
    error = tol + 1                                     # Initialize the error
    k = 1                                               # initialize the iteration
    
    while (error > tol) and (k <= max_iter): 
        x_new = T(x,model)                              # update by applying operator T
        error = np.max(np.abs(x_new-x))                 # update the error
        if k % print_step == 0:                         # the remainder of k/print_step is zero
            print(f"Completed iteration {k} with error {error}.") 
        x = x_new                                       # update x
        k += 1                                          # update the steps
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return x                                            # if successful, x is the fixed point


#--------------------------------------------------------------------------------------------#
#                             Bellman Operator in Job Search Model                           #
#--------------------------------------------------------------------------------------------#

def T (v,                                               # Value function
       model                                            # model parameters
      ):
    c, w_vals, n, β, ts_length=model                    # Unpack the model parameters
    return np.maximum(w_vals/(1-β), c + β * np.mean(v)) # return the value for each state



#--------------------------------------------------------------------------------------------#
#                                   v-greedy policy                                          #
#--------------------------------------------------------------------------------------------#

def get_greedy (v,                                      # Value function
                model                                   # model parameters
               ): 
    c, w_vals, n, β, ts_length=model                    # Unpack the model parameters
    σ= np.where(w_vals/(1-β)>=c + β * np.mean(v), 1, 0)
    return σ

#--------------------------------------------------------------------------------------------#
#                    Value Function Iteration Algorithm for Job Search Model                 #
#--------------------------------------------------------------------------------------------#

def value_function_iteration (model):
    c, w_vals, n, β, ts_length=model                    # Unpack the model parameters
    v_init = np.zeros(n)                                # initialize Value function
    v_star = successive_approx(T, v_init, model)        # SI to get v*
    σ_star = get_greedy(v_star, model)                  # σ* is v*-greedy
    return v_star, σ_star

# value_function_iteration(model)


#--------------------------------------------------------------------------------------------#
#                                CONTINUATION VALUE operator                                 #
#--------------------------------------------------------------------------------------------#

def g (h,                                              #input reservation wage
       model):                                         # Model parameters
    c, w_vals, n, β, ts_length = model                 # Input the model parameters
    return c + β * np.mean(np.maximum(w_vals/(1-β), h))
       
#--------------------------------------------------------------------------------------------#
#                             Compute CV directly algorithm                                  #
#--------------------------------------------------------------------------------------------#

def continuation_value_iteration (model):
    c, w_vals, n, β, ts_length = model                 # Input the model parameters
    h_init = 0                                         # inital guess of CV
    h_star = successive_approx(g, h_init, model)       # succesive approximation on g
    v_star = np.maximum(w_vals/(1-β),h_star)           
    return v_star, h_star
       
# continuation_value_iteration(model)