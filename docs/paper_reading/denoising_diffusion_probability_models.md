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

$$
\begin{aligned}
q(\mathbf x_t|\mathbf x_0) = \mathcal N(\mathbf x_t;\sqrt{\bar\alpha_t}\cdot \mathbf x_0, (1-\bar \alpha_t)\mathbf I)
\end{aligned}
$$

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
>    Where we require $\mu(\mathbf x_t)\to 0$ to occur, it must also be the case that $\sigma(\mathbf x_t) \to \mathbf I$. Employing a single coefficient to manage concurrent changes in both the mean and variance can enhance the robustness of the diffusion process. However, $\mu(\mathbf x_t)$ and $\sigma(\mathbf x_t)$ are not dimensionally equivalent, and variance is of a squared nature. As the original paper defines $\alpha_t$ directly for the variance, to maintain balance, we need to apply a square root operation to the mean.

![image-20231202180059377](https://qiniu.lianghao.work/image-20231202180059377.png)

#### 2. Reverse process and $L_{1:T-1}$

In the reverse diffusion process, the fundamental principle is that we have to eliminate the noise added in the forward process iteratively. This is accomplished by using a neural network model. In other words,  if we obtain the distribution $q(\mathbf x_{t-1}|\mathbf x_t)$,  we can progressively restore $\mathbf x_0$ from a standard normal distribution, denoted as $\mathbf x_T\sim \mathcal N(0, \mathbf I)$. *William showed that, for Gaussian (and binomial) distributions, the diffusion process's reversal has the same functional form as the forward process*. Therefore, in DDPM, the distribution of the reverse process, $q(\mathbf x_{t-1}|\mathbf x_t)$, is also a Gaussian distribution.  However, we cannot explicitly determine the analytic expression of $q(\mathbf x_{t-1}|\mathbf x_t)$.  Instead, it is common to employ a deep neural model, specifically a U-Net architecture paired with attention mechanisms, to infer the reverse distribution $p_\theta$.

<img src="https://learnopencv.com/wp-content/uploads/2023/02/denoising-diffusion-probabilistic-models-forward_and_backward_equations.png" style="zoom:67%;" />

1. The Markov chain for the reverse diffusion starts from where the forward process ends, i.e., at timestep $T$, where the data distribution has been converted into (nearly an) isotropic Gaussian distribution.

$$
\begin{aligned}
q(\mathbf x_T) &\approx \mathcal N(\mathbf x_t;0, \mathbf ) \\
p(\mathbf x_T) &:= \mathcal N(\mathbf x_t;0, \mathbf I) \\
\end{aligned}
$$

2. The PDF of the reverse diffusion process is an **"integral"** over all the possible pathways we can take to arrive at a data sample (in the same distribution as the original) starting from pure noise $\mathbf x_T$.

$$
\begin{aligned}
p_\theta(\mathbf x_0) &:= \int p_\theta(\mathbf x_{0:T})d\mathbf x_{1:T}\\
p_\theta(\mathbf x_{0:T}) &:= p(\mathbf x_T)\prod_{t=1}^T p_\theta(\mathbf x_{t-1}|\mathbf x_t)\\
p_\theta(\mathbf x_{t - 1}|\mathbf x_t) &:=\mathcal N(\mathbf x_{t-1};\mathbf \mu_\theta(\mathbf x_t, t), \Sigma_\theta(\mathbf x_t, t))\\
&:=\mathcal N(\mathbf x_{t-1};\mathbf \mu_\theta(\mathbf x_t, t), \sigma^2_t\mathbf I)\\
\end{aligned}
$$

Authors discuss their choices in $p_\theta(\mathbf x_{t-1}|\mathbf x_t) = \mathcal N(\mathbf x_{t-1};\mu_\theta(\mathbf x_t, t), \Sigma_\theta(\mathbf x_t, t)$ when $1< t\le T$.  First, they set $\Sigma_\theta(\mathbf x_t, t)=\sigma^2_t\mathbf I$​ to **untrained time-dependent constants**.  Experimentally, both $\sigma^2_t =\beta_t$ and $\sigma_t^2=\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ has **similar results**:

- $\sigma^2_t =\beta_t$ is optimal for $\mathbf x_0 \sim \mathcal N(\mathbf 0, \mathbf I)$
- $\sigma_t^2=\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ is optimal for $\mathbf x_0$ deterministically set to one point.

*There are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance*.

**Although we cannot obtain the inverse distribution $q(\mathbf x_{t-1}|\mathbf x_t)$, but if we know $\mathbf x_0$, we can obtain the $q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$ by the Bayesian formula**.

![image-20231202224025504](https://qiniu.lianghao.work/image-20231202224025504.png)

where $\mathbf{C}$ denotes the negligible terms that only include $\mathbf{x}_0$ and $\mathbf{x}_{t}$. The expression of Gaussian distribution is $\mathcal N(\mathbf x;\mu, \sigma^2) \propto \exp\left(-\frac{1}{2}(\frac{1}{\sigma^2}\mathbf x^2-\frac{2\mu}{\sigma^2}\mathbf x + \frac{\mu^2}{\sigma^2})\right)$.

$$
\left\{ \begin{array}{l}
\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{1}{\sigma^2}\\
\frac{2\sqrt\alpha_t}{1-\alpha_{t}}\mathbf x_t + \frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf x_0 = \frac{2\mu}{\sigma^2}
\end{array} \right.
$$

Now, we can obtain the parameter form of distribution $q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)$ as follows.

$$
\begin{aligned}
\tilde\mu_t(\mathbf x_t, \mathbf x_0) &=\mu=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf x_t + \frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1-\bar\alpha_t}\mathbf x_0\\
\tilde \beta_t &= \sigma^2=\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
\end{aligned}
$$

We combine Eq(12) and Eq(18), and we can obtain more specific form.

$$
\mu_\theta(\mathbf x_t, t)=\tilde\mu_t(\mathbf x_t, \mathbf x_0) = \frac{1}{\sqrt{\alpha_t}}(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t, t))
$$

**where $\epsilon_\theta(\mathbf{x}_t, t)$ is the Gaussian-distributed noise predicted by the deep neural network**. To sample $\mathbf x_{t-1}\sim p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ is to compute  

$$
\begin{aligned}
\mathbf x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t, t)) + \sigma_t\mathbf z, \text{ where } \mathbf z \sim \mathcal N(0, \mathbf I) 
\end{aligned}
$$

The complete sampling procedure, Algorithm 2, resembles Langevin dynamics with $\epsilon_θ$ as a learned gradient of the data
density.

![image-20231203122024793](https://qiniu.lianghao.work/image-20231203122024793.png)

**The reverse (or inference) process of the DDPM is outlined below**.

1. First, Gaussian noise $\epsilon_\theta(\mathbf x_t, t)$ is predicted at each timestep using  $\mathbf x_t$ and $t$ , from which we then determine the mean $\mu_\theta(\mathbf x_t, t)$.
2. Subsequently, the variance $\Sigma_\theta(\mathbf x_t, t)$ is calculated. 
3. Finally,  $\mathbf x_{t-1}$ is resampled from the posterior distribution  $q(\mathbf x_{t-1}|\mathbf x_t)$

#### 3. Loss function

![image-20231202124353163](https://qiniu.lianghao.work/image-20231202124353163.png)

The complete variational lower bound consists of $3$ parts, including $L_T, L_0, L_t$, where $1\le t\le T-1 $. 

1. $L_T=\mathbb E_q[D_{KL}(q(\mathbf x_T|\mathbf x_0)\|p(\mathbf x_T))]$: the forward diffusion process has no learnable parameters, and $\mathbf x_T$ is pure Gaussian noise, and therefore can be ignored.

2. $L_{t-1}=\mathbb E_q[D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)\| p_\theta(\mathbf x_{t-1}|\mathbf x_t)), \text{ where } 1\le t \le T-1]$: $q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)$ is the posterior distribution of the forward diffusion, and $p_\theta(\mathbf x_{t-1}|\mathbf x_t)$ is the posterior distribution of the reverse process. From the perspective of Variational Inference, $q(\cdot|\cdot)$ is the real posterior, and $p_\theta(\cdot |\cdot)$ is the variational posterior, so it can be obtained by deep network, And the two posterior are assumed to be **Gaussian distributions**, so the solution of KL divergence as follows.

$$
\begin{aligned}
L_{t-1} &=\mathbb E_q\left[ D_{KL}(\mathcal N_q(\mathbf x_{t-1}; {\tilde \mu_t}(\mathbf x_t, \mathbf x_0), \tilde \beta_t\mathbf I)\|\mathcal N_p(\mathbf x_{t-1}|\mu_\theta(\mathbf x_t, t), \sigma^2_t\mathbf I))\right]\\
&=\mathbb E_q\left[\frac{(\tilde \mu_t(\mathbf x_t, \mathbf x_0)-\mu_\theta(\mathbf x_t, t))^2}{2\sigma^2_t}+\frac{1}{2}\left(\frac{\tilde \beta_t}{\sigma^2_t}-1-\ln\frac{\tilde \beta_t}{\sigma_t^2}\right)\right]\\
&=\mathbb E_q\left[\frac{1}{2\sigma^2_t}\|(\tilde \mu_t(\mathbf x_t, \mathbf x_0)-\mu_\theta(\mathbf x_t, t))\|^2\right]+C(\tilde\beta_t, \sigma^2_t)\\
&=\mathbb E_q\left[\frac{1}{2\sigma^2_t\alpha_t}\|\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\bar\epsilon_t - \mathbf x_t +  \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf x_t, t)\|^2\right]+C(\tilde\beta_t, \sigma^2_t)\\
&=\mathbb E_q\left[\frac{\beta_t^2}{2\sigma^2_t\alpha_t(1-\bar\alpha_t)}\|\bar\epsilon_t - \epsilon_\theta(\mathbf x_t, t)\|^2\right]+C(\tilde\beta_t, \sigma^2_t)\\
&=\mathbb E_{\mathbf x_0,\bar\epsilon_t}\left[\frac{\beta_t^2}{2\sigma^2_t\alpha_t(1-\bar\alpha_t)}\|\bar\epsilon_t - \epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar \alpha_t}\cdot \epsilon, t)\|^2\right]+C(\tilde\beta_t, \sigma^2_t)\\
\end{aligned}
$$

3. $L_0=\mathbb E_q[-\log p_\theta(\mathbf x_0|\mathbf x_1)]$:  The authors of DDPM assume that the image consists of integers in $\{0, 1, \cdots,255\}$ scales linearly to $[-1, 1]$. **This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior $p(\mathbf x_T)$**. To obtain **discrete log-likelihoods**, they set this term to an **independent discrete decoder** derived from the Gaussian $\mathbf N(\mathbf x_0;\mu_\theta(\mathbf x_1, 1), \sigma^2_1\mathbf I)$. 

   where $D$ is the data dimensionality and the $i$​ superscript indicates extraction of one coordinate. *Similar to the discretized continuous distributions used in VAE decoders and autoregressive models, DDPM's choice here ensures that the variational bound is a lossless codelength of discrete data, without need of adding noise to the data or incorporating the Jacobian of the scaling operation into the log likelihood*. 

![image-20231203144517526](https://qiniu.lianghao.work/image-20231203144517526.png)

In ddpm, authors found it beneficial to sample quality (and simpler to implement) to train the following variant of the variational bound.

$$
\begin{aligned}
L_{\text{simple}}(\theta):=\mathbb E_{t, \mathbf x_0,\bar\epsilon_t}\left[\|\bar\epsilon_t - \epsilon_\theta(\sqrt{\bar\alpha_t}\cdot \mathbf x_0 + \sqrt{1-\bar\alpha_t}\cdot \epsilon, t)\|^2\right]
\end{aligned}
$$

where $t$ is uniform between 1 and $T$. The $t=1$ case corresponds to $L_0$ with the **integral in the discrete decoder** approximated by the **Gaussian probability density** function times the bin width, **ignoring $\sigma^2_1$ and edge effects**. The $t>1$ case corresponds to an unweighted version of $L_{t-1}$ ($L_T$ does not appear because the forward process variances $\beta_t$ are fixed) .

![image-20231203134530815](https://qiniu.lianghao.work/image-20231203134530815.png)

The simplified objective of $L_{\text{simple}}(\theta)$  discards the weighting in $L_{t-1}$, it is a weighted variational bound that **emphasizes different aspects of reconstruction** compared to the standard variational bound.

**The diffusion setup in DDPM causes the simplified objective to down-weight loss terms corresponding to small $t$**.  These terms train the network to denoise data with very **small amounts of noise**, so it is **beneficial** to down-weight them so that the **network can focus on more difficult denoising tasks at larger $t$ terms**. *This reweighting leads to better sample quality*.

<img src="https://qiniu.lianghao.work/image-20231203140652738.png" alt="image-20231203140652738" style="zoom: 80%;" />

### Experiments

**Setups** They set $T=1000$, and the forward process variances to constants increasing linearly from $\beta_1=10^{-4}$ to $\beta_T=0.02$. These constants were chosen to be small relative to data scaled to $[-1, 1]$, **ensuring that reverse and forward processes have approximately the same function form** while keeping the SNR at $\mathbf x_T$ as small as possible ($L_T=D_{KL}(q(\mathbf x_T|\mathbf x_0)||\mathcal N(0, \mathbf I))\approx 10^{-5}$ bits per dimension in DDPM's experiments).

**Deep Neural Network**: The DDPM uses a **U-Net backbone** similar to an **unmasked PixelCNN++** with **group normalization throughout** the **reverse process**. **Parameters are shared across time**, which is specified to the network using the **Transformer sinusoidal position embedding**. And DDPM uses **self.-attention** at the $16\times 16$ feature map resolution.  

![image-20231203140351842](https://qiniu.lianghao.work/image-20231203140351842.png)



![image-20231203141536363](https://qiniu.lianghao.work/image-20231203141536363.png)

![image-20231203142155071](https://qiniu.lianghao.work/image-20231203142155071.png)

![image-20231203142505515](https://qiniu.lianghao.work/image-20231203142505515.png)

![image-20231203142807239](https://qiniu.lianghao.work/image-20231203142807239.png)

![image-20231203142823495](https://qiniu.lianghao.work/image-20231203142823495.png)

### Reference

[1] [Understanding the diffusion model from the simple to the deep by user-ewrfcas](https://zhuanlan.zhihu.com/p/525106459)

[2] [An in-depth guide to denoising diffusion probabilistic models- from theory to implementation](https://learnopencv.com/denoising-diffusion-probabilistic-models)

[3] [diffusion probabilistic models from ZhenHu Zhang](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html)
