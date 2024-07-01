Here is a short description of all the code:

In this folder, it contains the code for

Dynamic Programming by John Stachurski and Thomas Sargent/ Chapter 5/
Applications/Optimal Inventories/

- book_inventory_mdp.py: this is the code from the book, the official code

- inventory_mdp.py: the first, original version of the code I write, extremely slow, takes around 2 hours to run and get the result

- njit_inventory_mdp.py: the improved version, using numba.njit decorator to improve the computation speeed. It takes around 2 seconds for all the results.

- njit_inventory_mdp_no_sa.py: another version of the numba improved code without using successive approximation, instead just change the value function iteration.


------------------

PLAN:

- jax_inventory_mdp.py: improved version using GOOGLE JAX

- njit_OPI_inventory_mdp.py: njit, not using VFI but using OPI

- njit_HPI_inventory_mdp.py: njit, not using VFI but using HPI

- jax_OPI_inventory_mdp.py: jax, not using VFI but using OPI

- jax_HPI_inventory_mdp.py: jax, not using VFI but using HPI


-------------------

numba.njit decorator note:

0. numba does not support scipy.stats.geom(), so we need to create a separate function for the pmf. But when using random variable, we can use np.random.geometric

1. numba does not support namedtuple very well, be careful, try not to use namedtuple other than unpack parameters

2. Be careful when using a list comprehension inside another function, although it looks more compact, it can trigger numba njit issues. When using list comprehension, we need to make sure we add np.array to make it into np.array type.

