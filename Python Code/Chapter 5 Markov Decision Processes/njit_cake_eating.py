#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#              DYNAMIC PROGRAMMING BY JOHN STACHURSKI AND THOMAS SARGENT                     #
#                                                                                            #
# This code is used for Chapter 5 Markov DECISION PROCESSES:                                 #
# Application: OPTIMAL SAVING WITH LABOR INCOME                                              #
# Improved computation efficiency using numba.njit                                           #
# Written by Longye Tian 02/07/2024                                                          #
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------#
#                               IMPORT LIBRARIES AND PACKAGES                                #
#--------------------------------------------------------------------------------------------#
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from numba import njit                                       
import time
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import LinearOperator
import quantecon as qe


# --------------------------------------------------------------------------------------------#
#                        USE NAMEDTUPLE TO STORE MODEL PARAMETERS                             #
# --------------------------------------------------------------------------------------------#

Optimal_Saving_MDP = namedtuple("saving_mdp", 
                                ("R",                          # gross interest rate
                                 "β",                          # discount rate
                                 "γ",                          # CRRA
                                 "W",                          # Wealth state space
                                 "w_size",
                                 "ρ",                          # Tauchen
                                 "ν",                          # Tauchen
                                 "m",                          # Tauchen
                                 "y_size"                      # income cardinality
                                ))


#---------------------------------------------------------------------------------------------#
#                       CREATE A FUNCTION TO INPUT MODEL PARAMETERS                           #
#---------------------------------------------------------------------------------------------#

def create_saving_mdp_model(R=1.01, 
                            β=0.98, 
                            γ=2.5, 
                            w_min=0.01, 
                            w_max=20.0, 
                            w_size=200, 
                            ρ=0.9, 
                            ν=0.1, 
                            m=3,
                            y_size=5):
    W = np.linspace(w_min, w_max, w_size)
    return Optimal_Saving_MDP(R=R, 
                              β=β, 
                              γ=γ, 
                              W=W,
                              w_size=w_size,
                              ρ=ρ, 
                              ν=ν, 
                              m=m,
                              y_size=y_size)

# saving_mdp = create_saving_mdp_model()


#---------------------------------------------------------------------------------------------#
#                                      NORMAL CDF                                             #
#---------------------------------------------------------------------------------------------#

@njit
def norm_cdf(x, mean=0, std=1):
    # Transform x to the standard normal
    z = (x - mean) / std
    
    # Use the Abramowitz & Stegun approximation for standard normal
    t = 1 / (1 + 0.2316419 * np.abs(z))
    d = 0.3989423 * np.exp(-z * z / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    return 1 - p if z > 0 else p



#---------------------------------------------------------------------------------------------#
#                                TAUCHEN DISCRETIZATION                                       #
#---------------------------------------------------------------------------------------------#

@njit
def Tauchen(saving_mdp):  
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    σ_y = np.sqrt(ν**2/(1-ρ**2))                               # W's std
    Y = np.linspace(-m*σ_y, m*σ_y, y_size)                     # State space by Tauchen
    s = (Y[y_size-1]-Y[0])/(y_size-1)                          # gap between two states
    Q = np.zeros((y_size,y_size))                              # Initialize Q
    for i in range(y_size):
        Q[i,0] = norm_cdf(Y[0]-ρ*Y[i]+s/2, std=σ_y)            
        Q[i,y_size-1] = 1 - norm_cdf(Y[y_size-1]-ρ*Y[i]-s/2, std=σ_y)   
        for j in range(1,y_size-1):
            Q[i,j] = norm_cdf(Y[j]-ρ*Y[i]+s/2, std=σ_y)-norm_cdf(Y[j]-ρ*Y[i]-s/2, std=σ_y)
    Y = np.exp(Y)
    return Y,Q


# saving_mdp = create_saving_mdp_model()
# Y,Q = Tauchen(saving_mdp)

#---------------------------------------------------------------------------------------------#
#                                   Utility Function                                          #
#---------------------------------------------------------------------------------------------#

@njit
def u(c, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    return ((c**(1-γ)))/(1-γ)

#--------------------------------------------------------------------------------------------#
#                                   BELLMAN EQUATION FOR V                                   #
#--------------------------------------------------------------------------------------------#
@njit
def B(v, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    
    W = np.reshape(W, (w_size, 1))
    WP = np.reshape(W, (1, w_size))
    c = W-(WP/R)

    
    return np.where(c>0, u(c, saving_mdp) + β * v, -np.inf)

# saving_mdp = create_saving_mdp_model()
# v = np.zeros((200,5))
# B(v, saving_mdp)
#σ = get_greedy(v, saving_mdp)

#---------------------------------------------------------------------------------------------#
#                                     Greedy Policy                                           #
#---------------------------------------------------------------------------------------------#

@njit
def get_greedy(v, saving_mdp):
    return np.argmax(B(v, saving_mdp), axis=-1)



#---------------------------------------------------------------------------------------------#
#                                   BELLMAN OPERATOR                                          #
#---------------------------------------------------------------------------------------------#

@njit
def T(v, saving_mdp):
    new_B = B(v, saving_mdp)
    w_size = new_B.shape[2]
    new_v = np.empty(new_B.shape[:2])
    for i in range(new_B.shape[0]):
            new_v[i] = np.max(new_B[i,:])
    return new_v




# saving_mdp = create_saving_mdp_model()
# v = np.zeros((200,5))
# T(v, saving_mdp)

    


#---------------------------------------------------------------------------------------------#
#                                SUCCESSIVE APPROXIMATION                                     #
#---------------------------------------------------------------------------------------------#

@njit
def successive_approx (T,                                        # A callable operator
                       v_init,                                   # Initial condition
                       saving_mdp,                               # Model parameter
                       tol = 1e-6,                               # Error tolerance
                       max_iter = 10_000,                        # max iterations
                       print_step = 25                           # Print at multiples of print_step
                      ):
    v = v_init                                                   # set the initial condition
    error = tol + 1                                              # Initialize the error
    k = 0                                                        # initialize the iteration
    
    while error > tol and k < max_iter: 
        new_v = T(v,saving_mdp)                                  # update by applying operator T
        error = np.max(np.abs(new_v-v))                          # update the error
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.") 
        v = new_v                                                # update x
        k += 1                                                   # update the steps
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return v

    
#--------------------------------------------------------------------------------------------#
#                                   POLICY OPERATOR                                          #
#--------------------------------------------------------------------------------------------#

# for loop to be compatible with numba
# The problem is in np.empty_like 
# If we initialize Σ = np.zeros((w_size, y_size))
@njit
def T_σ(v,σ,saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    
    # Σ = np.empty_like(σ)
    Σ = np.zeros((w_size,y_size))
    
    for i in range(w_size):
        for j in range(y_size):
           Σ[i,j] = W[σ[i,j]]
    W = np.reshape(W, (w_size, 1))
    Y = np.reshape(Y, (1, y_size))
    c = W + Y - (Σ/R)

    EV = np.empty((w_size, y_size))
    for i in np.arange(w_size):
        for j in np.arange(y_size):
            EV[i,j] = np.sum(np.array([v[σ[i,j],k] *  Q[j,k] for k in np.arange(y_size)]))

    return np.where(c>0, u(c, saving_mdp) + β * EV, -np.inf)


# multi-dimensional indexing without numba works

def T_σ_vec(v,σ,saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    Σ = W[σ]
    W = np.reshape(W, (w_size, 1))
    Y = np.reshape(Y, (1, y_size))
    c = W + Y - (Σ/R)

    V = v[σ]
    Q = np.reshape(Q, (1,y_size,y_size))
    EV = np.sum(V * Q, axis=-1)
    
    return np.where(c>0, u(c, saving_mdp) + β * EV, -np.inf)


#---------------------------------------------------------------------------------------------#
#                           COMPUTE REWARD AND REWARD OPERATOR  --- HPI                       #
#---------------------------------------------------------------------------------------------#

def compute_r_σ(σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)

    Σ = W[σ]
    W = np.reshape(W, (w_size, 1))
    Y = np.reshape(Y, (1, y_size))
    c = W + Y - (Σ/R)
    r_σ = np.where(c>0, u(c, saving_mdp), -np.inf)
    return r_σ


def R_σ(v, σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    σ = np.reshape(σ, (w_size, y_size,1))
    V = v[σ]
    Q = np.reshape(Q, (1,y_size,y_size))
    EV = np.sum(V * Q, axis=-1)
    return v - β * EV


#---------------------------------------------------------------------------------------------#
#                                  POLICY EVALUATION --HPI                                    #
#---------------------------------------------------------------------------------------------#

# use bicgstab to get value

def get_value_bicgstab(σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    r_σ = compute_r_σ(σ, saving_mdp)
    def _R_σ(v):
        return R_σ(v, σ, saving_mdp)
    
    A = LinearOperator((w_size * y_size, w_size * y_size), matvec=_R_σ)
    return bicgstab(A, r_σ)[0]


# use normal way to get value

def get_value(σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    Σ = W[σ]
    W = np.reshape(W, (w_size, 1))
    Y = np.reshape(Y, (1, y_size))
    c = W + Y - (Σ/R)
    r_σ = np.where(c>0, u(c, saving_mdp), -np.inf)
    x_size = w_size * y_size
    P_σ = np.zeros((w_size,y_size,w_size,y_size))
    for i in np.arange(w_size):
        for j in np.arange(y_size):
            for k in np.arange(y_size):
                P_σ[i,j,σ[i,j],k] = Q[j,k]

    r_σ = np.reshape(r_σ, (x_size,1))
    P_σ = np.reshape(P_σ, (x_size,x_size))
    I = np.eye(x_size)
    v_σ = np.linalg.solve((I-β*P_σ), r_σ)
    v_σ = np.reshape(v_σ, (w_size, y_size))
    
    return v_σ


# not working without using multi-indexing 
# solved, this is due to np.empty_like, avoiding using this function, just use np.zeros
@njit
def get_value_not_working_later_resolved(σ, saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)

    # not working part
    #Σ = np.empty_like(σ)
    #for i in range(w_size):
    #   for j in range(y_size):
    #      Σ[i,j] = W[σ[i,j]]

    # Solved: using np.zeros, will work
    Σ = np.zeros((w_size,y_size))
    for i in range(w_size):
      for j in range(y_size):
          Σ[i,j] = W[σ[i,j]]
    
    W = np.reshape(W, (w_size, 1))
    Y = np.reshape(Y, (1, y_size))
    c = W + Y - (Σ/R)
    r_σ = np.where(c>0, u(c, saving_mdp), -np.inf)
    x_size = w_size * y_size
    P_σ = np.zeros((w_size,y_size,w_size,y_size))
    for i in np.arange(w_size):
        for j in np.arange(y_size):
            for k in np.arange(y_size):
                P_σ[i,j,σ[i,j],k] = Q[j,k]

    r_σ = np.reshape(r_σ, (x_size,1))
    P_σ = np.reshape(P_σ, (x_size,x_size))
    I = np.eye(x_size)
    v_σ = np.linalg.solve((I-β*P_σ), r_σ)
    
    return np.reshape(v_σ, (w_size,y_size))
    
    
    
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#                                      ALGORITHMS                                             #
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------#
#                                VALUE FUNCTION ITERATION                                     #
#---------------------------------------------------------------------------------------------#

def value_function_iteration(saving_mdp):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    v_init = np.zeros(w_size, dtype=np.float64)
    v_star = successive_approx(T, v_init, saving_mdp)
    σ_star = get_greedy(v_star, saving_mdp)
    return v_star,  σ_star



#---------------------------------------------------------------------------------------------#
#                              OPTIMISTIC POLICY ITERATION                                    #
#---------------------------------------------------------------------------------------------#

# Currently OPI does not converge if we use T_σ (for loops indexing with numba)

# OPI converges if we use T_σ_vec (multi-dimensional indexing without numba)

def optimistic_policy_iteration(saving_mdp,
                                M=100,
                                tol=1e-6, 
                                max_iter=10_000,
                                print_step=25):
    
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    v = np.zeros((w_size, y_size))
    error = tol+1
    k = 0 

    while error > tol and k < max_iter:
        last_v = v
        σ = get_greedy(last_v,saving_mdp)
        for i in range(M):
            v = T_σ_vec(v, σ, saving_mdp)
        error = np.max(np.abs(last_v-v))
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.")
        k += 1
    if error <= tol:                                    
        print(f"Terminated successfully in {k} interations.")
        v_star_opi = v
        σ_star_opi = get_greedy(v_star_opi, saving_mdp)
    else:     
        print("Warning: hit iteration bound.")
    return v_star_opi, σ_star_opi


#---------------------------------------------------------------------------------------------#
#                                HOWARD POLICY ITERATIONS                                     #
#---------------------------------------------------------------------------------------------#

def howard_policy_iteration(saving_mdp, 
                            tol=1e-6, 
                            max_iter=10_000, 
                            print_step=25):
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    v = np.zeros((w_size, y_size))
    error = 1 + tol
    k=0
    while error > tol and k < max_iter:
        σ = get_greedy(v, saving_mdp)
        v_σ = get_value(σ, saving_mdp)
        error = np.max(np.abs(v_σ-v))
        v = v_σ
        if k % print_step == 0:                                   
            print(f"Completed iteration {k} with error {error}.")
        k += 1
    if error <= tol:
        print(f"Terminated successfully in {k} interations.")
        v_star_hpi = v
        σ_star_hpi = get_greedy(v_star_hpi, saving_mdp)
    else:
        print("Warning: hit iteration bound.")
    return v_star_hpi, σ_star_hpi



#---------------------------------------------------------------------------------------------#
#                          TIME SERIES AND DISTRIBUTION FOR WEALTH                            #
#---------------------------------------------------------------------------------------------#

def wealth_simulation(saving_mdp, 
                      w_init=0.01, 
                      ts_length=2000, 
                      random_state=0):
    
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    v_star, σ_star = value_function_iteration(saving_mdp)
    T = np.arange(ts_length)
    
    W_seq = np.zeros((ts_length,1))
    W_seq[0] = w_init
    for i in np.arange(1, ts_length):
        W_index = np.where(W==W_seq[i-1])[0][0]
        W_seq[i] = W[σ_star_opi[W_index, Y_index_seq[i-1]]] 
    
    fig, ax = plt.subplots()
    fig.suptitle('Wealth Simulation under optimal policy')
    ax.plot(T,W_seq, label='$w_t$')
    ax.legend()
    plt.show()


def wealth_distribution(saving_mdp, 
                        w_init=0.01, 
                        ts_length=2000, 
                        random_state=0):
    
    R, β, γ, W, w_size, ρ, ν, m, y_size = saving_mdp
    Y,Q = Tauchen(saving_mdp)
    v_star_opi, σ_star_opi = optimistic_policy_iteration(saving_mdp)
    T = np.arange(ts_length)

    mc = qe.MarkovChain(Q, state_values=Y)
    Y_seq = mc.simulate(ts_length=ts_length, random_state=random_state)
    mc_index = qe.MarkovChain(Q, state_values=np.arange(y_size))
    Y_index_seq = mc_index.simulate(ts_length=ts_length, random_state=random_state)
    
    W_seq = np.zeros((ts_length,1))
    W_seq[0] = w_init
    for i in np.arange(1, ts_length):
        W_index = np.where(W==W_seq[i-1])[0][0]
        W_seq[i] = W[σ_star_opi[W_index, Y_index_seq[i-1]]] 

    W_distribution = np.sort(W_seq,axis=0)
    fig, ax = plt.subplots()
    fig.suptitle('Wealth Simulation under optimal policy')
    ax.hist(W_distribution)
    ax.set_xlabel("wealth")
    plt.show()


#---------------------------------------------------------------------------------------------#
#                                         PLAYGROUND                                          #
#---------------------------------------------------------------------------------------------#

start_time = time.time()

#-------------ON YOUR MARKS---------SET----------------BANG!----------------------------------#


saving_mdp = create_saving_mdp_model()
#v_star_opi, σ_star_opi = optimistic_policy_iteration(saving_mdp)

optimistic_policy_iteration(saving_mdp)

#-------------------------------------------------------------------------------------------#
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
