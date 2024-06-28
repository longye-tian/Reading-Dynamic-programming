#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#               DYNAMIC PROGRAMMING BY JOHN STACHURSKI AND THOMAS SARGENT                    #
#                                                                                            #
# This code is used for Chapter 5 Markov DECISION PROCESSES:                                 #
# Application: OPTIMAL INVENTORY                                                             #
# Written by Longye Tian 28/06/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                               IMPORT LIBRARIES AND PACKAGES                                #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple
import quantecon as qe
from scipy.stats import geom
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------#
#                        CREATE A NAMEDTUPLE TO STORE MODEL PARAMETERS                       #
#--------------------------------------------------------------------------------------------#

Inventory_MDP = namedtuple("inventory_mdp", ("β",                  # DISCOUNT FACTOR
                                             "K",                  # STORAGE CAPACITY
                                             "x_vals",             # STORAGE STATE SPACE
                                             "c",                  # UNIT COST
                                             "κ",                  # FIXED COST
                                             "p",                  # DISTRIBUTION PARAMETER
                                             "d_max"               # MAXIMUM d
                                            ))



#--------------------------------------------------------------------------------------------#
#                       CREATE A FUNCTION TO INPUT MODEL PARAMETERS                          #
#--------------------------------------------------------------------------------------------#

def create_inventory_mdp_model(β=0.98, K=40, c=0.2, κ=2, p=0.6, d_max=100):
    x_vals = np.linspace(0,K,K+1)
    return Inventory_MDP(β=β, K=K, x_vals=x_vals, c=c, κ=κ, p=p, d_max=d_max)



#--------------------------------------------------------------------------------------------#
#                      BELLMAN EQUATION IN OPTIMAL INVENTORY MODEL                           #
#--------------------------------------------------------------------------------------------#

def B (x,                                                          # Initial state
       a,                                                          # Action
       v,                                                          # Value function
       inventory_mdp):
    β, K, x_vals, c, κ, p, d_max = inventory_mdp                   # UNPACK PARAMETERS
    
    D = np.linspace(1, d_max,d_max)                                # Demand process space
    X = x * np.ones(d_max)                                         # State value
    φ = ((1-p)**(D-1))*p                                           # Geometric distribution
    revenue = np.sum(np.minimum(X,D)*φ)                            # Revenue
    if a>0: 
        current_profit = revenue - c*a - κ
    else:
        current_profit = revenue -c*a