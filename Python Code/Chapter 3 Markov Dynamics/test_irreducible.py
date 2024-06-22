# Reference: Dynamic Programming by John Stachurski and Tom Sargent

# This code is used for Chapter 3 Markov Dynamics, Subsection 3.1.1.3. 

# Test whether a Markov matrix is irreducible or not

# by Longye Tian 22/06/2024



# Use quantecon packages

import quantecon as qe



# create a function with a matrix as input 

def test_irreducible(P):                  # P is the input Markov matrix
    mc = qe.MarkovChain(P)                # Create the correspoinding markov chain 
    return print(mc.is_irreducible)       # Print True or False as result


# Example

P = [[0.1, 0.9],
     [0.0, 1.0]]

# Result
test_irreducible(P)