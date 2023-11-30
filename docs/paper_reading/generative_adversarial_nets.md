## Generative Adversarial Nets
### [Paper](https://arxiv.org/abs/1406.2661)
The GAN simultaneously train two models:

- **Generative ($G$) Model** captures the **data distribution**
- **Discriminative ($D$) model** estimates the probability that a sample came from the training data rather than **$G$**

The training procedure for $G$ is to maximize the probability of $D$ making a mistake. 

In the space of arbitrary functions $G$ and $D$, a unique solution exists, with $G$ recovering the training data distribution and $D$ equal to $\frac{1}{2}$ everywhere. There is no need for any **Markov chains** or **unrolled approximate inference networks** during either training or generation of samples. 

To learn the generator's distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(z)$, then represent a mapping to data space as $G(z;\theta_g)$, where $G$ is a differentiable function represented by a multilayer perception with parameters $\theta_g$. We also define a second multilayer perception $D(x;\theta_d)$ that outputs a single scalar.  **$D(x)$ represents the probability that $x$ came from the data rather than $p_g$**. We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train G to minimize $\log (1 - D(G(z)))$. In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G, D)$:

$$
\mathop{\min}_{G}\mathop{\max}_D V(D,G) = \mathbb E_{x\sim p_{data}}[\log D(x)] + \mathbb E_{z\sim p_z(z)}[\log (1 - D(G(z)))]
$$

In practice, we alternate between $k$ steps of optimzing $D$ and one step of optimizing $G$. This results in $D$ being maintained near its optimal solution, so long as $G$ changes slowly enough.

![](https://qiniu.lianghao.work/image-20231130151552917.png)



The generator $G$ **implicitly** defines a probability distribution $p_g$ as the distribution of the samples $G(z)$ obtained when $z\sim p_z$

![image-20231130151923296](https://qiniu.lianghao.work/image-20231130151923296.png)



#### 1. Global Optimality of $p_g = p_{data}$, $x = g(z)$

We first consider the optimal discriminator $D$ for any given generator $G$.

##### Proposition 1. For G fixed, the optimal discriminator $D$ is

$$
D^*_{G}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

**The training criterion for the discriminator $D$, given any generator $G$, is to maximize the quantity $V(G, D)$**


$$
\begin{aligned}
V(G, D) &= \mathbb E_{x\sim p_{data}}[\log D(x)] + \mathbb E_{z\sim p_z(z)}[\log (1 - D(G(z)))]\\
&=\int_x p_{data}(x)\log(D(x)) + \int_z p_z(z) \log (1- D(g(z)))dz\\
&=\int_x p_{data}(x)\log (D(x)) + p_g(x) \log (1 - D(x))dx
\end{aligned}
$$


For any $(a, b) \in \mathbb R^2$, the function $y\rightarrow a\log(y) + b\log(1-y)$ achieves its maximum in $[0, 1]$ at $\frac{a}{a+b}$. 

Note that the training objective for $D$ can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y=y|x)$, where $Y$ indicates whether $x$ comes from $p_{data}$ (with $y=1$) or from $p_g$ (with $y=0$). The minimax game in Eq.1 can now be **reformulated** as :

$$
\begin{aligned}
C(G) &= \mathop{\max}_D V(G, D)\\
&= \mathbb E_{x\sim p_{data}}[\log D_G^*(x)] + \mathbb E_{z\sim p_z}[\log(1 - D_G^*(G(z)))]\\
&=\mathbb E_{x\sim p_{data}}[\log D_G^*(x)] + \mathbb E_{x\sim p_g}[\log(1 - D_G^*(x))]\\
&=\mathbb E_{x\sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb E_{x\sim p_g}\left[\log\frac{p_{g}(x)}{p_{data}(x) + p_g(x)}\right]
\end{aligned}
$$


**The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g = p_{data}$**. At that point, $C(G)$ achieved the value 

$$
\mathbf E_{x\sim p_{data}}[\log\frac{1}{2}] + \mathbb E_{x\in p_g}[\log\frac{1}{2}] = -\log 4
$$

and that by subtracting this expression from $C(G)=V(G, D)$, we obtain:

$$
\begin{aligned}
C(G)&=-\log 4 + KL(p_{data}||\frac{p_{data}+p_{g}}{2}) + KL(p_{g}||\frac{p_{data}+p_g}{2})\\
&=-\log 4  + 2\cdot JSD(p_{data}||p_g)
\end{aligned}
$$

> Since the Jensen–Shannon divergence between two distributions is always non-negative and zero only when they are equal, we have shown that $C^*=-\log 4$ is the global minimum of $C(G)$ and that the only solution is $p_g = p_{data}$, i.e., the generative model perfectly replicating the data generating process.

#### 2. Convergence of Algorithm 1

![image-20231130165911124](https://qiniu.lianghao.work/image-20231130165911124.png)

![image-20231130165953800](https://qiniu.lianghao.work/image-20231130165953800.png)

### [Reference](https://zhuanlan.zhihu.com/p/24767059)

| 文件/命令   |
| ----------- |
| [DataSet](../../source/dataset/generative/vae_celeba.py) |
|             |
|             |
|             |

