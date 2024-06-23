#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                Dynamic Programming by John Stachurski and Tom Sargent                      #
#                                                                                            #
#This code is used for Chapter 3 Markov Dynamics: Application: S-s dynamics                  #
#Written by Longye Tian 22/06/2024                                                           #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------#
#                                   Import Libraries and packages                            #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple
import quantecon as qe
from scipy.stats import geom
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------#
#                      Create a namedtuple to store model parameters                         #
#--------------------------------------------------------------------------------------------#

Inventory_model = namedtuple ("Inventory",("S",         # Order Size S
                                           "s",         # Reorder threshold
                                           "p",         # Geometric distribution parameter
                                           "ts_length", # Length of simulation
                                           "seed",      # seed for geometric distribution
                                           "D"          # Demand sequence
                                          ))

#--------------------------------------------------------------------------------------------#
#                               Create a model with parameter values                         #
#--------------------------------------------------------------------------------------------#
def create_inventory_model(S=100, 
                           s=10, 
                           p=0.4, 
                           ts_length=200, 
                           seed=1):    
    D = geom.rvs(p, size=ts_length, random_state=seed)  # Generate random demand     
    return Inventory_model(S=S,
                           s=s,
                           p=p, 
                           ts_length=ts_length, 
                           seed=seed, 
                           D=D) 

#--------------------------------------------------------------------------------------------#
#                       Create a function to generate simulated inventory sequence           #
#--------------------------------------------------------------------------------------------#
def simulate_inventory(inventory):
    S, s, p, ts_length, seed, D = inventory            # Unpack model parameters
    X = np.zeros(ts_length)                            # Initialize the Markov chain           
    X[1] = S                                           # Set initial conditions
    for t in range(1, ts_length-1):                    # Loop over time
        if X[t]<s:       
            X[t+1] = np.max(X[t]-D[t+1],0) + S         # Order S if below the threshold
        else:
            X[t+1] = np.max(X[t]-D[t+1])               # No order
    return X

#--------------------------------------------------------------------------------------------#
#                    Create a function to plot the simulated stock sequence                  #
#--------------------------------------------------------------------------------------------#
def plot_inventory(inventory, X): 
    S, s, p, ts_length, seed, D = inventory            # Unpack model parameters
    fig, ax = plt.subplots()                           # inititalize the fig and ax
    ax.plot(range(0,ts_length), X)                     # plot the simulated sequence
    ax.set_xlabel('time')                              # x-label
    ax.set_ylabel('Inventory Stock')                   # y-label
    plt.show()                                         # display the plot
    
# inventory = create_inventory_model()                 # Initialize the model
# X = simulate_inventory(inventory)                    # obtain the simulated sequence X
# plot_inventory(inventory, X)                         # Plot the simulated sequence


#--------------------------------------------------------------------------------------------#
#             Create a function to get the transition probabilities P(x,x')                  #
#--------------------------------------------------------------------------------------------#
def compute_mc(inventory, d_max=110):
    S, s, p, ts_length, seed, D = inventory            # Unpack model parameters
    n = S+s+1                                          # Dimension of the Transition matrix
    state_space = np.arange(0,n)                       # State space = {0,1,..., S+s}
    P = np.zeros((n,n))                                # Initialize the transition matrix
    h = np.zeros((1,d_max))                            # Initialize h
    for i in range(0,n):                               # for each state x_i
        for d in range(0,d_max):                       # Compute the possible future states               
            if i<=s:
                h[0,d] = np.max(i-d,0)+S
            else:
                h[0,d] = np.max(i-d,0)
        for j in range(0,n):                           # If future state = x_j
            for d in range(0,d_max):
                if h[0,d]==j:
                    P[i,j] += geom.pmf(d+1,p)          # add their pmf
    #mc = qe.MarkovChain(P, state_space)               # rowsum = 0.999 cannot use qe.MarkovChain
    return P


#--------------------------------------------------------------------------------------------#
#                  Create a function to get the stationary distribution                      #
#--------------------------------------------------------------------------------------------#

def compute_stat_dist(inventory, iteration = 10_000):
    P = compute_mc(inventory)                          # Get the transition matrix
    P_stat = P**iteration                              # Iterate the transition matrix
    ψ_stat = P_stat[0]                                 # The stationary distribution is the row of P
    return ψ_stat
                
## Problem: As P obtained before has rowsum = 0.999, iteration leads to 0


     