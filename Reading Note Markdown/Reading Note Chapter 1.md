## Definitions

**Definition (State)**

The state variable $X_t$ is a vector listing current values of variables relevant to choosing the current action.

**Definition (Value function)**

The value function $v_t(w)$ track the optimal lifetime payoffs from a given state at a given time. It depends only on the parameters (think about the *indirect utility function*). 

**Definition (Bellman equation)**

**Definition (Bellman's Principle of Optimality)**

**Definition (Globally Stable)**

A self-map $T$ is **globally stable** on $U\subset \mathbb{R}^n$ if 
- $T$ has a unique fixed point $x^*\in U$
- $T^ku\to u^*$ as $k\to\infty$ for all $u\in U$

**Definition (Invariant)**

Let $T$ be a self-map on $U\subset\mathbb{R}^n$. $C\subset U$ is **invariant** if 

$$\forall u\in C, Tu\in C$$

($T$ is a self-map on $C$ also.)

**Definition (Contraction)**
 
Let $T$ be a self-map on $U\subset\mathbb{R}^n, U\neq\emptyset$ with norm $\|\cdot\|$. $T$ is a contraction on $U$ if there exists $\lambda<1$ such that,

$$
\|Tu-Tv\| \le \lambda\|u-v\| \,\,\forall u,v\in U
$$

**Definition (Distribution)**

Given finite $X$, the set of distributions on $X$ is 

$$
\mathcal{D}(X):= \left\{\varphi\in\mathbb{R}^X:\varphi\ge 0,\,\,\sum_{x\in X}\varphi(x) = 1\right\}
$$

We say a random variable $Y$ has a distribution $\varphi\in\mathcal{D}(X)$ if

$$
\mathbb{P}\{Y=x\} = \varphi(x)\,\,\forall x\in X
$$

**Definition (Expectation)**
For $h\in\mathbb{R}^X and Y\sim \varphi$, the expectation of $h(Y)$ is

$$
\mathbb{E}h(Y):= \sum_{x\in X} h(x)\varphi(x) = \langle h,\varphi\rangle
$$

**Definition (Cumulative distribution function)**

If $X\subset \mathbb{R}$ (still finite), then,

$$
\Phi(x):= \mathbb{P}\{X\le x\} =\sum_{x'\in X} \mathbb{1}\{x'\le x\}\varphi(x')
$$
is the cumulative distribution function(CDF).

**Definition (Quantile) (Median)**

If $\tau\in[0,1]$, then the $\tau-th$ quantile of $Y$ is

$$
Q_tau Y:= \min\{x\in X: \Phi(x\ge \tau)\}
$$

Median is when $\tau = 1/2$.

**Definition**

Let $\{v_\sigma\}_{\sigma\in\Sigma}$ be a finite subset of $\mathbb{R}^X$. We set

$$
\bigwedge_\sigma v_\sigma (x):= \min_{\sigma\in\Sigma} v_\sigma(x)
$$

and

$$
\bigvee_\sigma v_\sigma(x):= \max_{\sigma\in\Sigma} v_\sigma(x)
$$


**Definition (Sublattice)**

Let $X$ be a finite set. A subset $V$ of $\mathbb{R}^X$ is called a sublattice of $\mathbb{R}^X$ if 

$$
u,v \in V \implies u\wedge v\in V, u\vee v \in V
$$




## Theorem

**Theorem**
The value function $v^*$ is the solution of the Bellman equation.

**Neumann Series Lemma**
Let $\rho(A)$ be the spetral radius of matrix A. If $\rho(A)<1$, then

- $I-A$ is nonsingular
- $\sum_{k\ge 0} A^k$ converges to $(I-A)^{-1}$

**Lemma**
For any square matrix $B$ and any matrix norm $\|\cdot\|$, we have

- $\rho(B)^k \le \|B^k\|$, $\forall k\in\mathbb{N}$

- $\rho(B) = \lim_{k\to\infty} \|B^k\|^{1/k}$  (Gelfand's Formula)

**Lemma**
If $\exists \overline u \in U, m\in\mathbb{N}$, s.t. $T^k u =\overline u$, $\forall u\in U$ and $k\ge m$, then $\overline u$ is the unique fixed point of $T$ in $U$.

**Lemma**

If

- $T$ is globally stable on $U\subset \mathbb{R}^n$ with fixed point $u^*$ and
- $C$ is nonempty, closed and invariant for $T$

then, $u^*\in C$


**Banach's Contraction Mapping Theorem**

If 

- $U$ is closed in $\mathbb{R}^n$
- $T$ is a contraction on $U$ wrt $\|\cdot\|$

then, there exists a unique fixed point $u^*\in U$ such that

$$
\|T^k-u^*\|\le\lambda^k\|u-u^*\|\,\,\forall k\in\mathbb{N}, u\in U
$$

In particular, $T$ is globally stable in $U$.


**Lemma**

If $f$ and $g$ are elements of $\mathbb{R}^X$, then

$$
|\max_{x\in X}f(x)-\max_{x\in X}g(x)|\le \max_{x\in X}|f(x)-g(x)|
$$


**Proposition**

Let 

- $V$ be a sublattice of $\mathbb{R}^X$
- $\{T_\sigma\}_{\sigma\in\Sigma}$ be a finite family of self-mapping on $V$

Set 

$$
Tv = \bigvee_{\sigma\in\Sigma}T_\sigma v \,\,\,(v\in V)
$$
By the sublattice property,  $T$ is a self-map on $V$.

**Lemma**

If $T_\sigma$ is a contraction of modulus $\lambda_\sigma$ w.r.t. $\|\cdot\|_{\infty}$ for each $\sigma\in\Sigma$, then $T$ is a contraction of modulus $\max_\sigma \lambda_\sigma$ under the same norm.

## Structure of a typical dynamic program

For a time period $t<T$, we (objective is to maximize the expected lifetime rewards **EPV**)

- observe the current state $X_t$

- choose an action $A_t$

- receive a reward $R_t(X_t,A_t)$

- update $X_{t+1} = F(X_t, A_t, \xi_{t+1})$

## Finite-Horizon Job Search Problem

This section use a finite-horizon job search problem to illustrate finite-period DP. In particular

- agent begins unemployed at time $t=0$

- receive a new job offer paying wage $W_t$ for all $t = 0,1, 2,\ldots, T$

- Two choices and corresponding rewards
  - *accept* $\implies$ work *permanetly* with wage at the time accepting the offer
  - *reject* $\implies$ receive constant unemployment compensation $c$ for the current period
  
The state variable is wage $W_t\sim \varphi\in \mathcal{D}(W)\,\,iid, W\subset \mathbb{R}_{+}$ finite and $\varphi$ is known. Action is to accept or reject the offer $A_{t}=0$ reject, $A_{t} = 1 $ accept.


We can represent this problem using Bellman Equations for each period $t = 0,1,2,\ldots, T$,i.e.,

$$
\begin{align}
v_t(w_t) &= \max\left\{\text{stopping value, continuation value}\right\}\\
&=\max\left\{\sum_{\tau = 0}^{T-t} \beta^\tau w_t, c + \beta \sum_{w'\in W} v_{t+1}(w')\varphi(w')\right\}
\end{align}
$$

We can solve for all $v_t$ by backward induction to calculate the reservation wage at each period. This solves the problem of whether to accept or reject the offer.

### Code of Finite-Horizon Job Search Problem (T period)

First, we start with importing `numpy` for numerical operations and `namedtuple` to store the model parameters.


```python
import numpy as np
from collections import namedtuple
```

A `namedtuple` is a convenient way to define a class. This type of data structure allows us to create tuple-like objects that have fields accessible by attribute lookup as well as being indexable and interable. 

In this model, we want to use the `namedtuple` to store values of the model, hence, we name it as `Model`. It requires the following parameters:

- `c`: the unemployment compensation
- `w_vals`: $W$, the finite wage space
- `n`: the cardinality of the wage space (in the following code, I use the uniform distribution, hence, this simplies the answer by not including extra parameters for the pdf)
- `β`: the discount factor


```python
Model = namedtuple("Model", ("c", "w_vals", "n", "β","T"))
```

Then we use a function to input specific values into the `namedtuple`, i.e.,


```python
def create_job_search_model(
    n = 50,          # wage grid size
    w_min = 11,      # lowest wage
    w_max = 60,      # highest wage
    c = 10,          # unemployment compensation
    β = 0.96,        # discount factor
    T = 10           # number of periods t= 0, 1,...,T
):
    """
    This function input the paramters with the above default values, and return a namedtuple
    """
    w_vals = np.linspace(w_min,w_max, n) # create a evenly spaced numbers over the specified intervals, with specified number of sample
    
    return Model(c = c, w_vals = w_vals, n = n, β = β, T=T) # return the namedtuple with the input parameters
```

Now we define a function that iteratively obtain the continuation value, and reservation wages


```python
def reservation_wage(model):
    c, w_vals, n, β, T = model.c, model.w_vals, model.n, model.β, model.T  # Input the model parameters
    H = np.zeros(T+1)  # Initialize the continuation value sequence
    R = np.zeros(T+1)  # Initialize the reservation wage sequence
    S = np.zeros((T+1,n))  # Initialize the maximum values at each state at each period
    H[T] = c         # Input the last continuation value which is just the unemployment compensation
    R[T] = c         # The reservation wage at the last period is just the unemployment compensation
    S[T,:] = np.maximum(c, w_vals) # At period T, it is just comparing the unemployment compensation with the wages
    for t in range(1, T+1):
        H[T-t] = c + β * np.mean(S[T-t+1,:]) # Assuming uniform distribution, we only need to calculate the mean
        df = np.geomspace(1, β**t, t+1)   # this generate the sequence for the denominator
        dfs = np.sum(df)  # this is the denominator for the reservation wage calculation
        R[T-t] = H[T+1-t]/dfs    # This calculate the reservation wage at time T-t
        S[T-t,:] = np.maximum(dfs * w_vals, H[T-t])   # This returns the maximum values for each wage state by comparing the continuation value and stopping value
    return R
    
```

This function iteratively generate the reservation wage sequence. We can show the result by create the model and use this function to calculate the reservation wage sequence.


```python
model = create_job_search_model()
reservation_wage(model)
```




    array([36.50032766, 35.38365172, 34.07207926, 32.50704612, 30.6067209 ,
           28.23780776, 25.19318638, 21.10849802, 15.29705719,  5.10204082,
           10.        ])



The key idea is to break down this multi-stage decision problem into a two-stage decision problem. We obtain the value functions by comparing the continuation value and the stopping value. 

## Infinite-Horizon Job Search Problem

The above example motivates the infinite-horizon job search problem. We let,

- $v^*(w)$ denote the maximum lifetime EPV for the wage offer $w$.

In the infinite horizon, we have

$$\text{Stopping value} = \frac{w}{1-\beta}$$

$$\text{Continuation value: }h^* = c+\beta\sum_{w'\in W}v^*(w')\varphi(w')$$\
This implies the optimal choice is
$$\mathbb{1}\{\text{Stopping value}\ge \text{Continuation value}\} = \mathbb{1}\left\{\frac{w}{1-\beta}\ge h^*\right\}$$

**Key Idea**
Solve the Bellman equation to obtain the value function $v^*$, the corresponding Bellman equation is

$$
v^*(w) = \max\left\{\dfrac{w}{1-\beta}, c+\beta\sum_{w'\in W}v^*(w')\varphi(w')\right\} \,\,\,\,(w\in W)
$$

### Solve the Value function $v^*$

We first introduce the **Bellman operator**, defined at $v\in\mathbb{R}^W$ by

$$
(Tv)(w) = \max\left\{\frac{w}{1-\beta}, c+\beta\sum_{w'\in W} v(w')\varphi(w')\right\} \,\,\,(w\in W)
$$

**Proposition**

$T$ is a contraction on $R^W$ with respect to $\|\cdot\|_{\infty}$.

This implies,

- there exists a unique fixed point $\overline v \in\mathbb{R}^W$ of $T$
- $T^kv\to \overline v$ as $k\to\infty$ for all $v\in \mathbb{R}^W$.
- $\overline v = v^*$

**Summary**

We can compute $v^*$ by successive approximation:
1. Choose any initial $v\in\mathbb{R}^W$
2. Iterate with $T$ to obtain $T^kv\approx v^*$ for some large $k$.

### Optimal policies

In general, for a dynamic program, choices means a sequence of actions $(A_t)_{t\ge 0}$, this specifies how agent will act at each $t$.

We assume $A_t$ depends on current and past events, i.e.,

$$
A_t = \sigma_t(X_t, A_{t-1}, X_{t-1}, A_{t-2},\ldots, A_0,X_0)
$$

In DP, $\sigma_t$ is called **policy function**.


**Key idea** Design the state variable $X_t$ such that 

- it is sufficient to determine the optimal current action
- but not so large as to be unmanagable


**Job search problem**:

- state variable is the current wage offer
- actions are to accept or reject the current wage offer, i.e., $A_t = \sigma(w)$
- a policy $\sigma: W\mapsto \{0,1\}$

Let $\Sigma$ be the set of all such maps.

**Definition ($v$-greedy policy)**

for each $v\in\mathbb{R}^W$, a $v$-greedy policy implies $\sigma \in \Sigma$ satisfying

$$
\sigma(w)=\mathbb{1}\left\{\frac{w}{1-\beta}\ge c+\beta\sum_{w'\in W} v(w')\varphi(w')\right\} 
$$

for all $w\in W$.

In this case, $\sigma$ is a $v$-greedy policy $\iff$ we accept the offer when the stopping value is greater than or equal to the continuation value **computed using $v$ **.


**Optimal choice/policy**

- agent adopt the $v^*$-greedy policy (**Bellman's principle of optimality**)

$$
\sigma^*(w) = \mathbb{1}\{w\ge w^*\}
$$
where $w^*:= (1-\beta) h^*$ is the reservation wage

#### Computation of Optimal policy using value function iteration

Since $T$ is globally stable on $\mathbb{R}^W$, we can compute an approximate optimal policy by **value function iteration**,i.e., 

1. applying successive approximation on $T$ to compute $v^*$
2. calculate $v^*-$greedy policy

## Value function iteration

We first create the function for successive approximate.

Choose an arbitraty starting point $u_0$ and apply $S$ iteratively to get the fixed point. 


```python
def successive_approx (
    S,                     # A callable operator
    x_0,                   # Initial condition, the arbitary starting point
    model,                 # Model parameters
    tol = 1e-6,            # Error tolerance used for approximation
    max_iter = 10_000,     # max interation to avoid infinite iteration due to not converging
    print_step = 25        # Print at multiples of print_step
):
    x = x_0                # set the initial condition
    error = tol + 1        # Initialize the error
    k = 1                  # initialize the interations
    
    while (error > tol) and (k <= max_iter): 
        x_new = S(x,model)       # update by applying operator T
        error = np.max(np.abs(x_new-x))  # the valuation of error is based on the norm
        if k % print_step == 0:  # the remainder of k/print_step is zero
            print(f"Completed iteration {k} with error {error}.") # This prints the error value if the steps is divisible by the print_step
        x = x_new         # assign the new value to x
        k += 1            # update the number of step by 1
    if error <= tol:      # After the iteration finishes and if error is small
        print(f"Terminated successfully in {k} interations.")
    else:     
        print("Warning: hit iteration bound.")
    return x              # if successful, x is the fixed point
```

Now we define the Bellman operator as discussed before


```python
def S (
    v,        # the value function
    model     # model parameters
):
    c, w_vals, n, β, T = model.c, model.w_vals, model.n, model.β, model.T  # Input the model parameters
    return np.maximum(w_vals/(1-β), c + β * np.mean(v))                    # return the value as state before
```

Then, we construct a function to obtain the $v-$greedy policy.


```python
def get_greedy (
    v,        # the value function, a np array
    model     # model parameters
): 
    c, w_vals, n, β, T = model.c, model.w_vals, model.n, model.β, model.T  # Input the model parameters
    σ= np.where(w_vals/(1-β) >= c + β * np.mean(v), 1, 0)                  # v-greedy
    return σ
```

Now we use value function iteration method to obtain the value function $v^*$ and optimal policy/$v^*$-greedy policy $\sigma^*$


```python
def value_function_iteration (model):
    c, w_vals, n, β, T = model.c, model.w_vals, model.n, model.β, model.T  # Input the model parameters
    v_init = np.zeros(n) # initialize the guess of value function
    v_star = successive_approx(S, v_init, model)
    σ_star = get_greedy(v_star, model)
    
    return v_star, σ_star
```


```python
value_function_iteration(model)
```

    Completed iteration 25 with error 0.03455934557837281.
    Completed iteration 50 with error 6.7009955273533706e-06.
    Terminated successfully in 57 interations.





    (array([1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1198.06629623, 1198.06629623, 1198.06629623,
            1198.06629623, 1200.        , 1225.        , 1250.        ,
            1275.        , 1300.        , 1325.        , 1350.        ,
            1375.        , 1400.        , 1425.        , 1450.        ,
            1475.        , 1500.        ]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1]))



## Method II: Compute the continuation value directly

Idea: compute the continuation value $h^*$ directly. This shifts the problem from $n$-dimension to one dimension. 

**Method**:

Recall, we have

$$
v^*(w) = \max\left\{\frac{w}{1-\beta}, c + \beta \sum_{w'\in W} v^*(w')\varphi(w')\right\}
$$

where $h^* = c + \beta \sum_{w'\in W} v^*(w')\varphi(w')$ this implies, 

$$
v^*(w') = \max\left\{\frac{w'}{1-\beta}, h^*\right\}
$$
Hence,

$$
h^* = c+\beta \sum_{w'\in W}v^*(w')\varphi(w') =c+\beta \sum_{w'\in W}\max\left\{\frac{w'}{1-\beta}, h^*\right\}\varphi(w')
$$

In a similar fashion, we introduce a mapping $g:\mathbb{R}_+\mapsto\mathbb{R}_+$,

$$
g(h) = c+\beta \sum_{w'\in W}\max\left\{\frac{w'}{1-\beta}, h\right\}\varphi(w')
$$

We can show that $g$ is a contraction on $\mathbb{R}_+$.


**New Algorithm**

1. compute $h^*$ via successive approximation on $g$ (iterate in $\mathbb{R}$ not in $\mathbb{R}^n$)

2. Optimal policy is 

$$
\sigma^*(w) = \mathbb{1}\left\{\frac{w}{1-\beta}\ge h^*\right\}
$$


```python

```
