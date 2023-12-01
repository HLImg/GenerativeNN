# Denoising Diffusion Probabilistic Models

![image-20231201155842769](https://qiniu.lianghao.work/image-20231201155842769.png)

The best results are obtained by training on a weighted variational bound designed according to a novel connection between **diffusion probabilistic models** and **denoising score matching with Langevin dynamics**. *DDPM naturally admits a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding.*

A diffusion model is a **parameterized Markow chain** trained using **variational inference** to produce samples matching the data after finite time. Transitions of this chain are learned to **reverse a diffusion process**, which is a **Markow chain** that **gradually adds noise** to the data in opposite direction of sampling until **signal is destroyed**.

> In this paper, the authors discussed that "despite their sample quality, the DDPM do not have competitive log-likelihoods compared to other likelihood-based models. They discovered that **the majority of DDPM's lossless codelengths are consumed to describe imperceptible image details.**"

### Preliminaries

Diffusion models are **latent variable models** of form $p_\theta(\mathbf x_0):=\int p_\theta(\mathbf x_{0:T})d\mathbf x_{1:T}$, where $\mathbf x_1, \cdots, \mathbf x_T$ are **latents** of the **same dimensionality** as the data $\mathbf x_0\sim q(\mathbf x_0)$. 

$$
\begin{aligned}
p_\theta(\mathbf x_{0:T}) &:= p(\mathbf x_T)\prod_{t=1}^T p_\theta(\mathbf x_{t-1}|\mathbf x_t)\\
p_\theta(\mathbf x_{t - 1}|\mathbf x_t) &:=\mathcal N(\mathbf x_{t-1};\mathbf \mu_\theta(\mathbf x_t, t), \Sigma_\theta(\mathbf x_t, t))
\end{aligned}
$$

The joint distribution $p_\theta(\mathbf x_{0:T})$ is called the **reverse process**, and it is defined as a Markov chain with learned Gaussian transitions starting ar $p(\mathbf x_T)=\mathcal N(\mathbf x_T;\mathbf 0, \mathbf I)$

What distinguished diffusion models from other types of latent variable models is that the **approximate posterior $q(\mathbf x_{1:T}|\mathbf x_0)$**, called the **forward process** or **diffusion process**, is fixed to a Markov chain that **gradually adds Gaussian noise to the data** according to a variance schedule $\beta_1, \cdots, \beta_T$:

$$
\begin{aligned}
q(\mathbf x_{1:T}|\mathbf x_0) &:=\prod_{t=1}^Tq(\mathbf x_t|\mathbf x_{t-1})\\
q(\mathbf x_t|\mathbf x_{t-1}) &:= \mathcal N(\mathbf x_t;\sqrt{1 - \beta_t}\cdot \mathbf x_{t-1}, \beta_t\mathbf I)
\end{aligned}
$$

Training is performed by optimizing the usual **variational bound** on **negative log-likelihood**.

$$
\begin{aligned}
\mathbb E[-\log p_\theta(\mathbf x_0)] &\le \mathbb E_q\left[-\log\frac{p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right]\\
&=\mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t\ge 1}\log\frac{p_\theta(\mathbf x_{t - 1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t - 1})}\right]\\
&=\mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t > 1}\log \frac{p_\theta(\mathbf x_{t - 1}| \mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t - 1})} - \log\frac{p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]
\end{aligned}
$$

The **forward process** variances $\beta_t$ can **be learned by reparameterization** or **held constant as hyperparameters**, and expressiveness of the **reverse process** is ensured in part by the choice of Gaussian conditionals in $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$**, because both processes have the same function form when $\beta_t$ are small**. ==A notable property of the forward process is that it admits sampling $\mathbf x_t$ at an arbitrary timestep $t$ in closed form: using the notation $\alpha_t := 1 - \beta_t$, and $\bar \alpha_t :=\prod_{s=1}^t \alpha_s$, we have==:
$$
\begin{aligned}
q(\mathbf x_t|\mathbf x_0) = \mathcal N(\mathbf x_t;\sqrt{\bar\alpha_t}\cdot \mathbf x_0, (1-\bar \alpha_t)\mathbf I)
\end{aligned}
$$
Efficient training is therefore possible by optimizing random terms of $L$ with stochastic gradient descent. Further improvements come from **variance reduction **.

According to the diffusion forward process, we can get $q(\mathbf x_t | \mathbf x_{t-1}, \mathbf x_0)$, and combined with the Bayesian ruler, we rewrite the **variational bound** $L$ as follows.


$$
\begin{aligned}
q(\mathbf x_t|\mathbf x_{t - 1}) &=q(\mathbf x_t|\mathbf x_{t - 1},\mathbf x_0)\\
&=\frac{q(\mathbf x_t, \mathbf x_{t - 1},\mathbf x_0)}{q(\mathbf x_{t - 1}, \mathbf x_0)}\\
&=\frac{q(\mathbf x_{t - 1}|\mathbf x_t, \mathbf x_0)q(\mathbf x_t|\mathbf x_0)q(\mathbf x_0)}{q(\mathbf x_{t - 1}|\mathbf x_0)q(\mathbf x_0)}\\
&=\frac{q(\mathbf x_{t - 1}|\mathbf x_t, \mathbf x_0)q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t - 1}|\mathbf x_0)}\\
\end{aligned}
$$

we eliminate terms that are irrelevant to the learned parameters $\theta$.

$$
\begin{aligned}
&L = \mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t > 1}\log \frac{p_\theta(\mathbf x_{t - 1}| \mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t - 1})} - \log\frac{p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\

&=\mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t > 1}\log \frac{p_\theta(\mathbf x_{t - 1}| \mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}\cdot \frac{q(\mathbf x_{t - 1}|\mathbf x_0)}{ q(\mathbf x_t|\mathbf x_0)} - \log\frac{p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q\left[-\log \frac{p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)} - \log q(\mathbf x_T|\mathbf x_0)-\sum_{t > 1}\log \frac{p_\theta(\mathbf x_{t - 1}| \mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)} - \log p_\theta(\mathbf x_0|\mathbf x_1) + \log q(\mathbf x_1|\mathbf x_0) \right]\\
&=\mathbb E_q\left[-\log \frac{p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)} -\sum_{t > 1}\log \frac{p_\theta(\mathbf x_{t - 1}| \mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)} - \log p_\theta(\mathbf x_0|\mathbf x_1) \right]\\
&=D_{KL}\left(q(\mathbf x_T|\mathbf x_0)\| p(\mathbf x_T)\right) + \mathbb E_q\left[\sum_{t > 1} D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)||p_\theta (\mathbf x_{t-1}|\mathbf x_t)) - \log p_\theta(\mathbf x_0|\mathbf x_1) \right]
\end{aligned}
$$

where the authors use KL divergence to directly compare $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ against **forward process posteriors**,which are **tractable** when conditioned on $\mathbf x_0$

$$
\begin{aligned}
q(\mathbf x_{t-1}|\mathbf x_{t}, \mathbf x_0) = \mathcal N(\mathbf x_{t-1}; {\tilde \mu_t}(\mathbf x_t, \mathbf x_0), \tilde \beta\mathbf I)
\end{aligned}
$$

where

$$
\begin{aligned}
\tilde \mu_t(\mathbf x_t, \mathbf x_0)&:=\frac{\sqrt{\bar\alpha_{t-1}}\cdot \beta_t}{1-\bar\alpha}\mathbf x_0 + \frac{\sqrt{\alpha_t}\cdot (1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf x_t \\
\tilde \beta_t &:= \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t
\end{aligned}
$$
