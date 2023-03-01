# ECE269 Linear Algebra and its Applications
# Mini Project 2 - Orthogonal Matching Pursuit

## Introduction

Given measurement model
$$
\mathbf{y} \ = \ \mathbf{Ax} \ + \ \mathbf{n}
$$
where $ \mathbf{y} \in \mathbb{R}^{M}$ is the (compressed, M < N) measurement, $ \mathbf{A} \in \mathbb{R}^{M \text{x} N}$ is the measurement matrix, and $\mathbf{n} \in \mathbb{R}^{M}$ is the additive noise. Here, $\mathbf{x} \in \mathbb{R}^{N}$ is the unknown signal (to be estimated) with $s ≪ N$ non-zero elements. The indices of the non-zero entries of $\mathbf{x}$ (also known as the support of $\mathbf{x}$) is denoted by $S = \{ i |x_{i} \neq 0\}$, with $|S| = s$.

Let $\mathbf{\hat{x}}$ is the estimate of $\mathbf{x}$ obtained from the OMP. To measure the performance of the OMP, we use the normalized error defined as
$$
\frac{||\mathbf{x} - \mathbf{\hat{x}}||_{2}}{||\mathbf{x}||_2}
$$
where $||\mathbf{x}||_2$ is the $L_2$ norm of $\mathbf{x}$. The average Normalized Error is obtained by averaging the Normalized Error over 2000 Monte Carlo runs.

### Experimental Setup

- Generate **A** as a random matrix with independent and identically distributed entries drawn from the standard normal distribution. Normalize the columns of **A**.
- Generate the sparse vector **x** with random support of cardinality $s$ (i.e. $s$ indices are generated randomly from integers $1 \text{to} N$), and the non-zero entries are drawn as uniform random variables in the range $[-10, -1] \bigcup [1, 10]$.
- The entries of noise **n** are drawn independently from the normal distribution with standard deviation $\sigma$ and zero mean.
- For each cardinality $s \in [1, s_{max}]$, the average Normalized Error should be computed by repeating steps 2000 times and averaging the results over these 2000 Monte Carlo runs.

## Implementation of OMP

The algorithm for OMP is as follows:
* Initialize a residual vector $\mathbf{r} = \mathbf{y}$ and an empty support set $\hat{\mathbf{S}} = \emptyset$.
* Calculate the maximum correlated vector $\mathbf{c} = \max{\mathbf{A}^T \mathbf{r}}$. This vector corresponds to the column of $\mathbf{A}$ that is most correlated with the residual vector $\mathbf{r}$. So, we can say that all correlated vectors obtained at the end of the algorithm should span ${\mathbf{y}}$.
* Add the index of the correlated vector (column of **A**) to the supported set to be estimated.
* Now we would solve the optimizing criteria mentioned below:
$$
\hat{\mathbf{v}}_{k} = \underset{\mathbf{v}}{\text{argmin}} ||\mathbf{y} - \mathbf{x'}||_2^2 
$$
where, $\mathbf{x'} \in span \{a_{i}| i \in \hat{\mathbf{S}}\}$, and $\hat{\mathbf{v}}_{k}$ will be the orthogonal projection of $\mathbf{y}$ onto the subspace spanned by the columns of $\mathbf{A}$ corresponding to the indices in $\hat{\mathbf{S}}$.
* Update the residual vector $\mathbf{r} = \mathbf{y} - \hat{\mathbf{v}}_{k}$.
* When the algorithm converges (we will terminate the loop using the condition mentioned below), the support set $\hat{\mathbf{S}}$ will contain the indices of the non-zero entries of $\mathbf{x}$. And the estimate of $\mathbf{x}$ will be $\hat{\mathbf{x}}$, which is defined as $\hat{\mathbf{x}}_k = \hat{\mathbf{\alpha}}_{i}$ (where $k =\hat{\mathbf{S}}_{i}$).
* From previously mentioned point we can say that $\mathbf{y}$ is spanned by columns of $\mathbf{A}$ with coefficients (solution of normal equation, $\hat{\mathbf{\alpha}}$) obtained in the $\hat{\mathbf{x}}$.
  
### Solving the optimizing criteria

The optimizing criteria can be solved using the following steps:
* As we know, $\mathbf{x'} \in span \{a_{i}| i \in \hat{\mathbf{S}}\}$, so we write $\mathbf{x'} = \Sigma^{k}_i \alpha_i a_i$; where $a_i$ are the columns of $\mathbf{A}$ which has maximum correlation with residual vector as mentioned above. Hence, our goal is to find the values of $\alpha_i$ to satisfy given optimality condition.
* We would differentiate the above equation with respect to $\alpha_i$ and equate it to zero to find the optimal values of $\alpha_i$.
* We would get the following equation:
$$
y^{T}a_i = (\Sigma^{k}_{j}\alpha_j a_j)^{T}a_i \\
y^{T}a_i = (\Sigma^{k}_{j}\alpha_j a_j^{T}a_i)
$$
We got the below equation by transposing terms on both sides of equation. Since $a_j^{T}a_i$ is a scalar, we can write it as $a_i^{T}a_j$ and then we can write the above equation as:
$$
a_i^{T}y = (\Sigma^{k}_{j}\alpha_j a_i^{T}a_j)
$$
* To solve the above equation we will write the equation in matrix form:
$$
\begin{bmatrix}
y^{T}a_1 & y^{T}a_2 & \cdots & y^{T}a_k
\end{bmatrix}^{T}
=\begin{bmatrix}
a_1^{T}a_1 & a_1^{T}a_2 & \cdots & a_1^{T}a_k \\
a_2^{T}a_1 & a_2^{T}a_2 & \cdots & a_2^{T}a_k \\
\vdots & \vdots & \ddots & \vdots \\
a_k^{T}a_1 & a_k^{T}a_2 & \cdots & a_k^{T}a_k
\end{bmatrix} 
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
\vdots \\
\alpha_k
\end{bmatrix} \\
\begin{bmatrix}
y^{T}a_1 & y^{T}a_2 & \cdots & y^{T}a_k
\end{bmatrix}^{T}
=\begin{bmatrix}
a_1^{T} \\
a_2^{T} \\
\vdots \\
a_k^{T}
\end{bmatrix}
\begin{bmatrix}
a_1 & a_2 & \cdots & a_k
\end{bmatrix}
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
\vdots \\
\alpha_k
\end{bmatrix} \\
$$
Now let $\hat{\mathbf{A}} = \begin{bmatrix}a_1 & a_2 & \cdots & a_k\end{bmatrix}$ and $\hat{\mathbf{\alpha}} = \begin{bmatrix}\alpha_1 & \alpha_2 & \cdots & \alpha_k\end{bmatrix}^{T}$.
* Above equation can be written as:
$$
\hat{\mathbf{A}}^{T}y = \hat{\mathbf{A}}^{T}\hat{\mathbf{A}}\hat{\mathbf{\alpha}}
$$
* Solution to optimality condition is given by:
$$
\hat{\mathbf{\alpha}} = (\hat{\mathbf{A}}^{T}\hat{\mathbf{A}})^{-1}\hat{\mathbf{A}}^{T}y
$$
* The matrix $(\hat{\mathbf{A}}^{T}\hat{\mathbf{A}})^{-1}\hat{\mathbf{A}}^{T}$ is called pseudoinverse $({\mathbf{A}^{\dagger}})$ of $\hat{\mathbf{A}}$.
* We now have $\mathbf{x'} = \hat{\mathbf{A}}\hat{\mathbf{\alpha}}$ to satisfy the optimality condition. Hence, solution to the optimality condition is given by $\hat{\mathbf{v}}_{k} = \hat{\mathbf{A}}\hat{\mathbf{\alpha}}$.
  
### Termination criteria of OMP

The mentioned OMP algorithm can be terminated using one of the conditions mentioned below:
- When the product $\mathbf{A}^{T}\mathbf{r}$ which is correlation measure is less than a threshold value (here used as 0.1).
- When the error $||y-\mathbf{A}\hat{\mathbf{x}}||_2$ is less than a threshold value (here used as $10^{-3}$).
- When the error $||y-\mathbf{A}\hat{\mathbf{x}}||_2$ is less than a set noise threshold value (denoted as $||\mathbf{n}||_{2}$).
  
## Noiseless case: (n = 0)

Here $ \mathbf{n} = 0$, so $ \mathbf{y} = \mathbf{Ax}$. We will stop OMP iterations when $ ||\mathbf{y} - \mathbf{A}\mathbf{\hat{x}}||_2 < 10^{-3}$. The probability of Exact Support Recovery (ESR) is defined as the probability that the support of $\mathbf{\hat{x}}$ is exactly the same as the support of $\mathbf{x}$ i.e. $S = \hat{S}$ by averaging over 2000 Monte Carlo runs (2000 random realizations of $\mathbf{A}$).

### N = 20
