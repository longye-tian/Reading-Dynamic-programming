#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                Dynamic Programming by John Stachurski and Tom Sargent                      #
#                                                                                            #
# This code is used for Chapter 3 Markov Dynamics: Application: Day Laborer                  #
# Written by Longye Tian 22/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                                         Imports                                            #
#--------------------------------------------------------------------------------------------#

import numpy as np
from collections import namedtuple
import quantecon as qe
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------#
#                      Create a namedtuple to store model parameters                         #
#--------------------------------------------------------------------------------------------#

Day_Laborer_Model = namedtuple("day_laborer",            # Name
                              ("α",                      # P(x_{t+1}=2|x_t=1)
                               "β",                      # P(x_{t+1}=1|x_t=2)
                               "seed"                    # Random_state parameter
                              ))
#--------------------------------------------------------------------------------------------#
#                               Create the model function                                    #
#--------------------------------------------------------------------------------------------#

def create_day_laborer_model(α=0.3,β=0.2,seed=1):
    return Day_Laborer_Model(α=α, β=β, seed=seed)


#--------------------------------------------------------------------------------------------#
#                               Create a function to Update the State                        #
#--------------------------------------------------------------------------------------------#

def update_day_laborer(init_state,                     # Initial state
                       day_laborer):                   # Model parameters
    α,β,seed=day_laborer                               # Unpack Model parameters
    if init_state==1 :
        next_state = bernoulli.rvs(α,                  # init_state = 1
                                   size=1, 
                                   random_state=seed)+1
    elif init_state==2 :                               # init_state = 1
        next_state = bernoulli.rvs(1-β, 
                                   size=1, 
                                   random_state=seed)+1
    else:                                              # wrong input
        print("Invalid Initial State")

    return next_state


#--------------------------------------------------------------------------------------------#
#                            Compute Stationary Distribution                                 #
#--------------------------------------------------------------------------------------------#

def compute_stationary_dist(day_laborer):
    α,β,seed=day_laborer                               # Unpack Model parameters
    P = [[1-α,α],
         [β,1-β]]
    mc = qe.MarkovChain(P,("Unemployed","Employed"))
    ψ_stationary = mc.stationary_distributions[0]
    return ψ_stationary


#--------------------------------------------------------------------------------------------#
#                            Compute the sequence  ψP^t for different ψ                      #
#--------------------------------------------------------------------------------------------#

def compute_dist_sequence(day_laborer, 
                          init_dist,
                          ts_length):
    α,β,seed=day_laborer                               # Unpack Model parameters
    P = [[1-α,α],
         [β,1-β]]
    Ψ = np.zeros((ts_length,len(P)))
    for t in range(0,ts_length):
        Ψ[t,:] = np.matmul(init_dist, np.linalg.matrix_power(P,t))
    return Ψ

# compute_dist_sequence(day_laborer, [0.1,0.9], 10)

#--------------------------------------------------------------------------------------------#
#                          Ergodicity Property Check                                         #
#--------------------------------------------------------------------------------------------#

def time_average(day_laborer,
            init_state,
            ts_length):
    α,β,seed=day_laborer                               # Unpack Model parameters
    P = [[1-α,α],
         [β,1-β]]
    mc = qe.MarkovChain(P,("Unemployed","Employed"))
    simulated_sequence = mc.simulate(ts_length, init=init_state, random_state=seed)
    time_average = [ (1/ts_length)* np.count_nonzero(simulated_sequence == "Unemployed",(1/ts_length)*np.count_nonzero(simulated_sequence == "Employed")]
    return time_average
# day_laborer = create_day_laborer_model()
# time_average(day_laborer, "Unemployed", 10000)