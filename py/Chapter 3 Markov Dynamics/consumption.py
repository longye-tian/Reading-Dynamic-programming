#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#                DYNAMIC PROGRAMMING BY JOHN STARCHURSKI AND THOMAS SARGENT                  #
#                                                                                            #
# This code is used for Chapter 3 Markov Dynamics: Application: Valuing Consumption Streams  #
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

Consumption_Model = namedtuple("consumption",("n",              # Tauchen 
                                              "m",              # Tauchen
                                              "β",              # discount factor
                                              "ρ",              # AR 1 coefficent
                                              "ν",             
                                              "γ"
                                             ))

def create_consumption_model (n=25, m=3, β=0.98,ρ=0.96, ν=0.05, γ=2):
    return Consumption_Model(n=n, m=m,β=β,ρ=ρ, ν=ν, γ=γ)


#--------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                      #
#--------------------------------------------------------------------------------------------#
def Tauchen(consumption):  
    n, m, β, ρ, ν, γ = consumption                             # Unpack model parameters
    σ_x = np.sqrt(ν**2/(1-ρ**2))                               # X's std
    X = np.linspace(-m*σ_x, m*σ_x, n)                          # State space by Tauchen
    s = (X[n-1]-X[0])/(n-1)                                    # gap between two states
    P = np.zeros((n,n))                                        # Initialize P
    for i in range(n):
        P[i,0] = norm.cdf(X[0]-ρ*X[i]+s/2, scale=σ_x)          # j=1
        P[i,n-1] = 1 - norm.cdf(X[n-1]-ρ*X[i]-s/2, scale=σ_x)  # j=n
        for j in range(1,n-1):
            P[i,j] = norm.cdf(X[j]-ρ*X[i]+s/2, scale=σ_x)-norm.cdf(X[j]-ρ*X[i]-s/2, scale=σ_x)
    return X,P
    

# consumption = create_consumption_model()
# X,P = Tauchen(consumption)
    
#--------------------------------------------------------------------------------------------#
#                            CONSTRUCT EPV AS A FUNCTION OF X                                #
#--------------------------------------------------------------------------------------------#

def EPV(consumption):
    n, m, β, ρ, ν, γ = consumption                             # Unpack model parameters
    X,P = Tauchen(consumption)                                 # Tauchen
    c = np.exp(X)                                              # Consumption stream
    r = (c**(1-γ)/(1-γ))                                       # Reward stream
    v = np.linalg.solve((np.identity(n)-β*P), r)               # Solve for v
    return v,c,r

# v = EPV(consumption)

#--------------------------------------------------------------------------------------------#
#                                       PLOT v                                               #
#--------------------------------------------------------------------------------------------#

def plot_EPV(consumption):
    n, m, β, ρ, ν, γ = consumption                             # Unpack model parameters
    v,c,r = EPV(consumption)                                   # EPV
    fig, ax = plt.subplots()                                   # Initialize the plot
    ax.plot(X, v, label="value path")
    ax.set_xlabel('State')                                     # x-label
    ax.set_ylabel('EPV')                                       # y-label
    ax.legend()                                                # legend                      
    plt.show()                                                 
    
# plot_EPV(consumption)
    
    