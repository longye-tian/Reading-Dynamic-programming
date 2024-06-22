# Reference: Dynamic Programming by John Stachurski and Tom Sargent
# This code is used for Chapter 3 Markov Dynamics, Subsection 3.1.1.2. Application: S-s dynamics
# by Longye Tian 22/06/2024


# Imports

import numpy as np
from collections import namedtuple
import quantecon as qe
from scipy.stats import geom
import matplotlib.pyplot as plt


# Create a namedtuple to store model parameters

Inventory_model = namedtuple ("Inventory",("S",         # Order Size S
                                           "s",         # Reorder threshold
                                           "p",         # Geometric distribution parameter
                                           "ts_length", # Length of simulation
                                           "seed",      # seed for geometric distribution
                                           "D"          # Demand sequence
                                          ))


# Create a model with parameter values

def create_inventory_model(S=100, s=10, p=0.4, ts_length=200, seed=1):
    D = geom.rvs(p, size=ts_length, random_state=seed)  # Generate random demand process under geometric distribution
    return Inventory_model(S=S,s=s,p=p, ts_length=ts_length, seed=seed, D=D) 


# Create a function to generate simulated inventory sequence
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


# Create a function to plot the simulated stock sequence

def plot_inventory(inventory, X): 
    S, s, p, ts_length, seed, D = inventory            # Unpack model parameters
    fig, ax = plt.subplots()                           # inititalize the fig and ax
    ax.plot(range(0,ts_length), X)                     # plot the simulated sequence
    ax.set_xlabel('time')                              # x-label
    ax.set_ylabel('Inventory Stock')                   # y-label
    plt.show()                                         # display the plot
    
inventory = create_inventory_model()                   # Initialize the model
X = simulate_inventory(inventory)                      # obtain the simulated sequence X
plot_inventory(inventory, X)                           # Plot the simulated sequence
