# Dynamic Programming Reading Group Plan

## Week 1: Introduction (June 10 - June 16, 2024)
- **Reading:** Chapter 1 (Pages 1-41)
- Key Concept:
  - For $a,b,c \in\mathbb{R}$, $|a\vee c - b\vee c|\le |a-b|$
  - Basic Dynamic Programming Algorithm
  - Banach's Contraction Mapping Theorem and Proof
  - Neumann Series Lemma, Gelfand's formula
  - Finite vs. Infinite job search model
  - Algorithm for Value Function Iteration
- Python Code
  - Create namedtuple, use function to create model
  - Successive approximation
  - Greedy function
  - value function iteration
  - direct iteration 
  - See details in job_search.py

## Week 2: Operators and Fixed Points (June 17 - June 23, 2024)
- **Reading:** Chapter 2 (Pages 42-80)
- Key Concepts:
  - $T$ is globally stable and $T$ dominates $S$ implies the unique fixed point of $T$ dominates any fixed point of $S$.
  - Hartman-Grobman Theorem
  - Convergence rate and related proof
  - Newton's fixed point method
  - Partial order, order-preserving, property of sublattice, stochastic dominance
  - Linear operator, equivalence with matrix, computation advantage
  - Perron-Frobenius Theorem and Lemma
  - Flatten a matrix
  - Markov operator and its matrix representation

## Week 3: Markov Dynamics (June 24 - June 30, 2024)
- **Reading:** Chapter 3 (Pages 81-104)
- Key Concept:
   - Markov chain, stationarity, irreducibility, ergodicity, monotonicity
   - Approximation: Tauchen discretization
   - Conditional Expectation, LIE, LTP
- Code:
   - QuantEcon MarkovChain packages: MarkovChain, Stationary distribution, simulate
   - Ergodicity: calculate time average and compare to stationary distribution.
   - See: inventory_simulation.py; day_laborer.py

## Week 4: Optimal Stopping (July 1 - July 7, 2024)
- **Reading:** Chapter 4 (Pages 105-127)
- EXERCISE 4.1.13.
- **Learning Objectives:**
  - Comprehend the concept of optimal stopping and its use in decision-making.
  - Explore examples of optimal stopping in firm valuation with exit.
  - Understand the role of continuation values in optimal stopping problems.

## Week 5: Markov Decision Processes (MDPs) (July 8 - July 14, 2024)
- **Reading:** Chapter 5 (Pages 128-178)
- **Learning Objectives:**
  - Define Markov decision processes and identify their key components.
  - Apply MDPs to problems like optimal inventories and savings with labor income.
  - Learn about Q-factors and their use in dynamic programming.

## Week 6: Stochastic Discounting (July 15 - July 21, 2024)
- **Reading:** Chapter 6 (Pages 181-211)
- **Learning Objectives:**
  - Understand the concept of stochastic discounting and its implications for valuation.
  - Learn about the spectral radius condition and its testing methods.
  - Apply MDPs with state-dependent discounting in inventory management.

## Week 7: Nonlinear Valuation (July 22 - July 28, 2024)
- **Reading:** Chapter 7 (Pages 212-244)
- **Learning Objectives:**
  - Explore the significance of moving beyond contraction maps in dynamic programming.
  - Understand the impact of recursive preferences on optimal savings and risk-sensitive preferences.
  - Learn about Epstein-Zin preferences and their role in dynamic programming.

## Week 8: Recursive Decision Processes (RDPs) (July 29 - August 4, 2024)
- **Reading:** Chapter 8 (Pages 245-290)
- **Learning Objectives:**
  - Define recursive decision processes and understand their properties.
  - Differentiate between contracting and eventually contracting RDPs.
  - Explore applications of RDPs in risk-sensitive decision-making and adversarial agents.

## Week 9: Abstract Dynamic Programming (August 5 - August 11, 2024)
- **Reading:** Chapter 9 (Pages 291-305)
- **Learning Objectives:**
  - Understand abstract dynamic programs and their generalization of dynamic programming.
  - Learn about max-optimality and min-optimality in abstract dynamic programs.
  - Relate abstract dynamic programs to recursive decision processes.

## Week 10: Continuous Time (August 12 - August 18, 2024)
- **Reading:** Chapter 10 (Pages 306-337)
- **Learning Objectives:**
  - Understand the basics of continuous time Markov chains and their application in dynamic programming.
  - Learn about continuous time Markov decision processes and their construction.
  - Explore the application of continuous time models to job search problems.
