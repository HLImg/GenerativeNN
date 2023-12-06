# Denoising Diffuison Implicit Models

![image-20231206161139555](https://qiniu.lianghao.work/image-20231206161139555.png)

Denoising diffusion probabilistic models (DDPMs) have achieved high-quality image generation without adversarial training, yet they require simulating a Markov chain for many steps in order to produce a sample. The difference between DDPM and DDIM is as follows.

- DDPM: the generative process is defined as the **reverse** of a particular **Markovian** diffusion process
- DDIM:  generalizes DDPMs via a class of **non-Markovian** diffusion processes. These non-Markovian processes can correspond to generative processes that are **deterministic**, giving rise to implicit models that produce high-quality samples much faster.

GANs exhibit higher sample quality than likelihood-based methods such as variational autoencoders, autoregressive models, and normalizing flows. However, GANs require very specific choices in optimization and architectures in order to stabilize training, and **could fail to cover modes of data distribution**.

Recently works on iterative generative models have demonstrated the ability to produce samples comparable to that of GANs, without having to perform adversarial training.  A **critical drawback** of these models is that they require many iterations to produce a high-quality sample. For DDPMs, this is because the generative process (from noise to data) approximates the reverse of the forward *diffusion process*, which could have thousands of steps; iterating over all the steps is required to produce a single sample.

**DDIMs** are **implicit probabilistic models** and are closely related to DDPMs, in the sense that they are trained with the same objective function.

## Variational Inference for Non-Markovian Forward Process

The key observation from DDIM is that the DDPM objective in the form of  $L_\gamma$ only depends on the marginals $q(x_t|x_0)$, but not directly on the joint $q(x_{1:T}|x_0)$

$$
\begin{aligned}
L_\gamma(\epsilon_\theta) :=\sum_{t=1}^T\mathbb E_{x_0\sim q(x_0), \epsilon_t\sim\mathcal N(0, I)}\left[\|\epsilon_\theta^{(t)}(\sqrt{\alpha_t}\cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon_t\|^2_2\right]\gamma_t
\end{aligned}
$$

### Non-Markovian Forward Processes

Let us consider a family $\mathcal Q$ of inference distributions, indexed by a real vector $\sigma \in \mathbb R^T_{\ge 0}$:

$$
\begin{aligned}
q_\sigma( x_{1:T}|x_0) := q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)
\end{aligned}
$$

where $q_\sigma (x_T|x_0)=\mathcal N(\sqrt{\alpha_T}\cdot x_0, (1-\alpha_T)I)$, and for all $t>1$

$$
q_\sigma(x_{t-1}|x_t, x_0) = \mathcal N(\sqrt{\alpha_{t-1}}\cdot x_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\alpha_0}\cdot x_0}{\sqrt{1-\alpha_t}},\sigma^2I)
$$

The mean function is chosen in order to ensure that $q_\sigma(x_t|x_0)=\mathcal N(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)$ for all $t$.

As we have shown in the [ddpm](https://github.com/HLImg/GenerativeNN/blob/master/docs/paper_reading/denoising_diffusion_probability_models.md), the **forward process** can be derived from Bayes' relu:

$$
q_\sigma(x_t|x_{t-1}, x_0)=\frac{q_\sigma(x_{t-1}|x_t, x_0)q_\sigma(x_t|x_0)}{q_\sigma(x_{t-1}|x_0)}
$$

Unlike the diffusion process in DDPM, the forward process here is no longer Markovian, since each $x_t$ could depend on both $x_{t-1}$ and $x_0$.  The magnitude of $\sigma$ controls how stochastic the forward process is; when $\sigma\to 0$, we reach an extreme case where as long as we observe $x_0$ and $x_t$ for some $t$, then $x_{t-1}$ becomes known and fixed.

### Generative Process and Unified Variational Inference Objective

Intuitively, given a noisy observation $x_t$, the first step is to make a prediction of the corresponding $x_0$, and then use it to obtain a sample $x_{t-1}$ through the reverse conditional distribution $q_\sigma(x_{t-1}|x_t, x_0)$.

By rewriting the normal forward diffusion process, one can then predict the denoised observation, which is a **prediction of $x_0$** given $x_t$.

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon, \text{ where } \epsilon\sim \mathcal N(0, I)\\
f_\theta^{(t)}(x_t)&:=(x_t -\sqrt{1 - \alpha_t}\cdot \epsilon^{(t)}_\theta(x_t))/\sqrt{\alpha_t}
\end{aligned}
$$

Then, it defines the generative process with a fixed prior $p_\theta(x_T)=\mathcal N(0, I)$ and 

$$
p_\theta^{(t)}(x_{t-1}|x_t)=\left\{ \begin{array}{l}
\mathcal N(f_\theta^{(1)(x_1)},  \sigma^2_1\mathbf I), \text{ if } t=1\\
q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)), \text{otherwise}
\end{array} \right.
$$

The authors add some Gaussian noise (with covariance $\sigma^2_1 I$) for the case of $t=1$ to ensure that the generative process is supported everywhere. 

$$
\begin{aligned}
p_{\theta}(x_{0:T})&:=p_\theta(x_T)\prod_{t=1}^Tp_\theta^{(t)}(x_{t-1}|x_t)\\
q_\sigma(x_{1:T}|x_0)&:=q_\sigma(x_T|x_0)\prod_{t=2}^Tq_\sigma(x_{t-1}|x_t. x_0)
\end{aligned}
$$

The parameters $\theta$ are optimized following variational inference objective (which is a functional over $\epsilon_\theta$)

$$
\begin{aligned}
\begin{aligned}
J_\sigma(\epsilon_\theta)&:=\mathbb E_{x_{0:T}\sim q_\sigma(x_{0:T})}[\log q_\sigma(x_{1:T}|x_0) - \log p_\theta(x_{0:T})]\\
&=\mathbb E_{x_{0:T}\sim q_\sigma(x_{0:T})}\left[\log \left(q_\sigma(x_T|x_0)\prod_{t=2}^Tq_\sigma(x_{t-1}|x_t, x_0)\right) -\log \left(p_\theta(x_T)\prod_{t=1}^Tp_\theta^{(t)}(x_{t-1}|x_t)\right)\right]\\
&=\mathbb E_{x_{0:T}\sim q_\sigma(x_{0:T})}\left[\log q_\sigma(x_T|x_0) + \sum_{t=2}^T\log q_\sigma(x_{t-1}|x_t, x_0)-\sum_{t=1}^T\log p_\theta^{(t)}(x_{t-1}|x_t)-\log p_\theta(x_T)\right]\\
&=\mathbb E_{x_{0:T}\sim q_\sigma(x_{0:T})}\left[\sum_{t=2}^T\log \frac{q_\sigma(x_{t-1}|x_t, x_0)}{p_\theta^{(t)}(x_{t-1}|x_t)}+\log \frac{q_\sigma(x_T|x_0)}{p_\theta(x_T)}-\log p_\theta^{(1)}(x_{0}|x_1)\right]\\
&=\mathbb E_{x_{0:T}\sim q_\sigma(x_{0:T})}\left[D_{KL}(q_\sigma(x_T|x_0)||p_\theta(x_T))+ \sum_{t=2}^T D_{KL}(q_\sigma(x_{t-1}|x_t, x_0)||p_\theta^{(t)}(x_{t-1}|x_t))-\log p_\theta^{(1)}(x_{0}|x_1)\right]
\end{aligned}
\end{aligned}
$$

From the definition of $J_\sigma$, it would appear that a different model has to be trained for every choice of $\sigma$, **since it corresponds to a different variational objective (and a different generative process)**. However, $J_\sigma$ is equivalent to $L_\gamma$ for certain weights $\gamma$ .

> For all $\sigma > 0$, there exists $\gamma \in \mathbb R^T_{>0}$ and $C \in \mathbb R$, such that $J_\sigma = L_\gamma +C$
>
> The variational objective $L_\gamma$ is special in the sense that if parameters $\theta$ of the models $\epsilon_\theta^{(t)}$ are not shared across different $t$, then the optimal solution for $\epsilon_\theta$ will not depend on the weights $\gamma$ (as global optimum is achieved by separately maximizing each term in the sum). 
>
> This property of $L_\gamma$ has two implications. On the one hand, this justified the use of $L_1$ as a **surrogate objective function** for the variational lower bound in DDPMs; On the other hand, **since $J_\sigma$ is equivalent to some $L_\gamma$, the optimal solution of $J_\sigma$ is also the same as that $L_1$**. **Therefore, if parameters are not shared across $t$ in the model $\epsilon_\theta$, then the $L_1$ objective can be used as a surrogate objective for the variational objective $J_\sigma$ as well.**


where

$$
\begin{aligned}
q_\sigma(x_t|x_0)&=\mathcal{N}(\sqrt{\alpha_t}\cdot x_0, (1-\alpha_t)I) \\
p_\theta^{(t)}(x_{t-1}|x_t)&=q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) \\
&=\mathcal{N}\left(\sqrt{\alpha_{t-1}}\cdot f_\theta^{(t)}(x_t)+\sqrt{1-\alpha_{t-1}-\sigma^2_t}\cdot \frac{x_t - \sqrt{\alpha_t}\cdot f_\theta^{(t)}(x_t)}{\sqrt{1-\alpha_t}}, \sigma^2_t I\right)
\end{aligned}
$$



![image-20231206171155411](https://qiniu.lianghao.work/image-20231206171155411.png)

## Sampling from Generalized Generative Processes

### Denoising Diffusion Implicit Models

![image-20231206171334038](https://qiniu.lianghao.work/image-20231206171334038.png)

where $\epsilon_t\sim \mathcal N(0, I)$ is standard Gaussian noise independent of $x_t$, and **different choices of the hyper-parameter $\sigma$ values results in different generative processes**, all while using the same model $\epsilon_\theta$, re-training the model is unnecessary.
$$
\sigma = \eta\sqrt{(1-\alpha_{t-1}/（1-\alpha_t)}\sqrt{1-\alpha_t/\alpha_{t-1}}
$$


- $\eta=0$:  Given $x_{t-1}$ and $x_0$,  the forward process becomes deterministic , except for $t=1$.And the reverse process will lose randomness. That is to say, the generative image from a single noise is determined by fixing the sampling steps.
- $\eta=1$ for all $t$: the reverse process of DDIM is the same as DDPM's. 

![image-20231206174410121](https://qiniu.lianghao.work/image-20231206174410121.png)

### Accelerated Generation Processes

The denoising objective $L_1$ does not depend on the specific forward procedure as long as $q_\sigma(x_t|x_0)$ is fixed. DDPM considers forward processes with lengths smaller than $T$, which accelerates the corresponding generative processes without having to train a different model.

**DDIM considers the forward process as defined not on all the latent variables $x_{1:T}$, but on a subset $\{x_{\tau_1}, x_{\tau_2},\cdots, x_{\tau_S}\}$, where $\tau$ is an increasing sub-sequence of $[1, 2, \cdots, T]$**. In particular, DDIM defines the sequential forward process over $x_{\tau_1}, \cdots, x_{\tau_S}$ such that $q(x_{\tau_i}|x_0)=\mathcal N(\sqrt{\alpha_{\tau_i}}x_0, (1-\alpha_{\tau_i})I)$ matches the "marginals".

![image-20231206175656323](https://qiniu.lianghao.work/image-20231206175656323.png)

The generative process now samples latent variables according to reversed ($\tau$), which we term (sampling) trajectory. When the length of the sampling trajectory is much smaller than $T$, we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.
