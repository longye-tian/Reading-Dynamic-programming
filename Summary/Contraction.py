# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Summary on Contraction

# %% [markdown]
# ### Definition (Contraction)
#
# Let $U$ be a nonempty subset of $\mathbb{R}^n$ and let $\|\cdot\|$ be a norm on $\mathbb{R}^n$. 
#
# A self map $T$ on $U$ is called a **contraction** on $U$ with respect to $\|\cdot\|$ if there exists a $\lambda<1$, such that,
#
# $$
# \|Tu-Tv\|\le \lambda\|u-v\|
# $$
#
# for all $u,v\in U$.
#
# The constant $\lambda$ is called the **modulus of contraction**.

# %% [markdown]
# ### Definition (Eventually contracting)
#
# Fix $U\subset \mathbb{R}^X$. We call $T:U\mapsto U$ **eventually contracting** if there exists a $k\in\mathbb{N}$ and a norm $\|\cdot\|$ on $\mathbb{R}^X$ such that $T^k$ is a contraction on $U$ under $\|\cdot\|$.

# %% [markdown]
# ## Properties of Contraction

# %% [markdown]
# 1. Contraction is **continuous** and has **at most one** fixed point.
#
# 2. In Banach space, contraction is globally stable with unique fixed point.
#
# 3. Contraction converges **at least linearly**.
#
# 4. The **upper envelope** of contractions is a contraction with modulus as the maximum of the contractions.
#
# 5. In a **closed** set (Banach space), **eventually contracting** implies globally stable.
#
# 6. $T$ is a contraction wrt to one norm **does not implies** $T$ is a contraction wrt to another norm.
#
# 7. $T$ is eventually contracting with one norm implies $T$ is eventually contracting **on every norm**.

# %% [markdown]
# ## Useful Theorem and Propositions to prove contraction

# %% [markdown]
# #### Proposition (Absolute difference of maximum with other is less than the absolute difference)
#
# $$
# |\alpha \vee x - \alpha\vee  y |\le |x-y| \tag{$\alpha, x, y\in\mathbb{R}$}
# $$
#
# This proposition is useful when proving Bellman operator is a contraction. 

# %% [markdown]
# #### Proposition (Absolute difference of maximum with other is less than the absolute maximum of difference)
#
# $$
# |\max_{z\in D} f(z)- \max_{z\in D} g(z)| \le \max_{z\in D} |f(z)-g(z)|
# $$

# %% [markdown]
# #### Lemma: Blackwell's condition
#
# Let $T: U\mapsto U$ be an order-preserving self-map, and there exists $\beta\in (0,1)$ such that,
#
# $$
# T(u+c)\le Tu + \beta c
# $$
#
# for all $u\in U$, $c\in\mathbb{R}_+$.
#
# Then, $T$ is a contraction of modulus $\beta$ on $U$ wrt the supremum norm. 
#
# #### Proposition: Generalized Blackwell Condition
#
# Let $T:U\mapsto U$ be an order-preserving self-map. If there exists $L\in\mathcal{L}_+(\mathbb{R}^X)$ with $\rho(L)<1$, and
#
# $$
# T(v+c)\le Tv+Lc 
# $$
#
# for all $c,v\in\mathbb{R}^X$ with $c\ge 0$.
#
# Then $T$ is **eventually contracting** on $U$.

# %% [markdown]
# #### Proposition (Bounded by a positive linear operator implies eventually contracting)
#
# Let $T: U\mapsto U$, $U\subset \mathbb{R}^X$. If there exists a positive linear operator $L$ on $\mathbb{R}^X$ such that $\rho(L)<1$ and
#
# $$
# |Tv-Tw|\le L|v-w|
# $$
#
# for all $v,w\in U$, then $T$ is an **eventual contraction** on $U$.
