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

where the authors use KL divergence to directly compare $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ against **forward process posteriors**, which are **tractable** when conditioned on $\mathbf x_0$

$$
\begin{aligned}
q(\mathbf x_{t-1}|\mathbf x_{t}, \mathbf x_0) = \mathcal N(\mathbf x_{t-1}; {\tilde \mu_t}(\mathbf x_t, \mathbf x_0), \tilde \beta_t\mathbf I)
\end{aligned}
$$

where

$$
\begin{aligned}
\tilde \mu_t(\mathbf x_t, \mathbf x_0)&:=\frac{\sqrt{\bar\alpha_{t-1}}\cdot \beta_t}{1-\bar\alpha}\mathbf x_0 + \frac{\sqrt{\alpha_t}\cdot (1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf x_t \\
\tilde \beta_t &:= \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t
\end{aligned}
$$

### Diffusion models and denoising autoencoders

Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. **One must choose the variances $\beta_t$ of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process**.

![image-20231202124353163](https://qiniu.lianghao.work/image-20231202124353163.png)

#### 1. Forward process and $L_T$

> The authors ignore the fact that the forward process variances $\beta_t$ are learnable by reparameterization and instead fix them as constants.  Thus,  in the implementation of DDPM, the approximate posterior $q$ has no learnable parameters, so $L_T$ is a constant in during training and can be ignored.

<img src="https://learnopencv.com/wp-content/uploads/2023/02/denoising-diffusion-probabilistic-models_forward_process_changing_distribution-768x493.png" style="zoom:67%;" />

> 1. In the forward diffusion process, we slowly and iteratively add noise to (corrupt) the images in our train set such that they "move out or move away" from their existing subspace.
> 2. What we are doing here is converting the unknown and complex distribution that our training set belongs to into one that is easy for us to sample a (data) point from and understand.
> 3. At the end of the forward process, the images become entirely unrecognizable. The complex data distribution is wholly transformed into a (chosen) simple distribution. Each image gets mapped to a space outside the data subspace.

The probability distribution function (PDF) used to define **the Forward Diffusion Process** is a **Normal or Gaussian distribution**, the mean is $(1-\beta_t)\mathbf x_{t-1}$, and the covariance is $\beta_t\mathbf I$.  

The term $\mathbf \beta$ is known as the **diffusion rate** and is precalculated using a **variance scheduler**. The term $\mathbf I$  is an **identity matrix**. Therefore, the distribution at each time step is called **isotropic Gaussian**. *By choosing sufficiently large timesteps and defining a well-behaved schedule of $\beta_t$,*  the repeated application of the forward diffusion process **converts** the **data distribution** to be nearly an **isotropic Gaussian distribution**.

##### Reparameterization trick

In variational autoencoders,  we encode the input images to infer two latent parameters: the mean $\mu_{\theta}$ and the variance $\sigma^2_{\theta}$. We then use the decoder to reconstruct image $\mathbf x$ by sampling from the normal distribution $\mathcal N(\mu_\theta, \sigma^2_\theta)$. However, due to the stochastic nature of direct sampling, gradient backpropagation is not straightforward. We commonly employ the reparameterization trick, which can be formulated as follows.

$$
\begin{aligned}
\mathbf x = \mu_{\theta} + \sigma_\theta \cdot \epsilon, \ \epsilon \sim \mathcal N(0, \mathbf I)
\end{aligned}
$$

##### Random timestep, $\mathbf x_t$ can be represented by $\mathbf x_0$ and $\beta_t$

We can easily sample image $\mathbf x_t$ from a normal distribution according to reparameterized as :

$$
\begin{aligned}
\mathbf x_t &= \mathbf x_{t-1}\sqrt{1-\beta_t} + \sqrt{\beta_t}\cdot\epsilon_t\\
&\text{where}\  \epsilon_t \sim \mathcal N(0, \mathbf I)
\end{aligned}
$$

If we use $\alpha_t=1-\beta_t, \bar \alpha_t  = \prod_{i=1}^t\alpha_i$, then the $\mathbf x_t$ can be formulated as follows.

$$
\begin{aligned}
\mathbf x_t = \mathbf x_{t-1} \sqrt {\alpha_t} + \sqrt{1-\alpha_t}\cdot \epsilon_t
\end{aligned}
$$

We set $t \in [1, t]$, we can get some examples as follows. 

$$
\begin{aligned}
\mathbf x_1 &= \mathbf x_0 \sqrt{\alpha_1} + \sqrt{1-\alpha_1}\cdot\epsilon_0\\
\mathbf x_2 &= \mathbf x_0 \sqrt{\alpha_1\alpha_2} + \sqrt{(1-\alpha_1)\alpha_2}\cdot \epsilon_1 + \sqrt{1-\alpha_2}\cdot\epsilon_2\\
&=\mathbf x_0 \sqrt{\alpha_1\alpha_2} + \sqrt{1-\alpha_1\alpha_2}\cdot \bar\epsilon_2, \ \ \bar \epsilon_2\sim \mathcal N(0, \mathbf I)\\
&\cdots\\
\mathbf x_t  &= \mathbf x_0\prod_{i=1}^t\sqrt{\alpha_i}+\sqrt{1 - \prod_{i=1}^t\alpha_i}\cdot \bar\epsilon_t\\
\mathbf x_t &= \mathbf x_0\sqrt{\bar \alpha_t} + \sqrt{1-\bar\alpha_t}\cdot \bar\epsilon_t
\end{aligned}
$$

where $\sqrt{(1-\alpha_1)\alpha_2}\cdot \epsilon \sim \mathcal N(0, \mathbf I(1-\alpha_1)\alpha_2)$, $\sqrt{1-\alpha_2}\cdot \epsilon \sim \mathcal N(0, (1-\alpha_2)\mathbf I)$ , the addition between the two follows Gaussian dirstribution $\mathcal N(0, (1-\alpha_1\alpha_2)\mathbf I)$. 

**Using the above formulation, we can sample at any arbitrary timestep $t$ in Markoc chain**.

> 1. **The efficiency of hyper-parameter $\beta_t$?**
>
>    The forward diffusion process hopes that $\mathbf x_T$ will gradually approach the **standard norm distribution** $\mathcal N(0,  \mathbf I)$ when $T\rightarrow  +\infty$. And $\mathbf x_t$ is only related to its previous step $\mathbf x_{t-1}$. $\mathbf x_t$ and $\mathbf x_{t-1}$ are **linear Gaussian changes**. The relationship between $\mu(\mathbf x_t)$ and $\mathbf x_{t-1}$ must be a linear relationship multiplied by a certain coefficient $\mathbf x_{t-1}$.
>
>    **In practice, the authors of DDPMs use a "linear variance scheduler" and define $\beta$ in the range $[0.0001, 0.02]$ and set the timesteps $T=1000$**. *Diffusion models scale down the data with each forward process step (by a $\sqrt{1-\beta_t}$ factor) so that variance does not grow when adding noise*.
>
> 2. **Why $\alpha_t=1-\beta_t$ is less than $1$ ?**
>
>    If $\alpha_i < 1$, then $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ will converge to $0$  as $t$ increases, which then leads to $\mu(\mathbf x_t)\rightarrow 0$.
>
> 3. **Why should there be a square root applied to the coefficient of the mean? $\sqrt{\mathbf x_0\alpha_t}$**
>
>    Where we require $\mu(\mathbf x_t)\to 0$ to occur, it must also be the case that $\sigma(\mathbf x_t) \to \mathbf I$. Employing a single coefficient to manage concurrent changes in both the mean and variance can enhance the robustness of the diffusion process. However, $\mu(\mathbf x_t)$ and $\sigma(\mathbf x_t)$ are not dimensionally equivalent, and variance is of a squared nature. As the original paper defines $\alpha_t$ directly for the variance, to maintain balance, we need to apply a square root operation to the mean

![image-20231202180059377](https://qiniu.lianghao.work/image-20231202180059377.png)

### Reference

[1] [Understanding the diffusion model from the simple to the deep by user-ewrfcas](https://zhuanlan.zhihu.com/p/525106459)

[2] [An in-depth guide to denoising diffusion probabilistic models- from theory to implementation](https://learnopencv.com/denoising-diffusion-probabilistic-models)

[3] [diffusion probabilistic models from ZhenHu Zhang](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html)
