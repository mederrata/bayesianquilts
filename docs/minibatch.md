

Let $q(w|\theta)$ be a joint probability distribution on parameters $w$ conditional on internal parameters $\theta$. The negative ELBO is:


$$
\begin{align}
    L(\xi|D) &= D_{KL}(q(\theta|\xi)|P(\theta)) - \mathbb{E}_q \log P(D| \theta) \nonumber \\
    &= D_{KL}(q(\theta|\xi)|P(\theta)) - \sum_{n=1}^N \mathbb{E}_q \log P(D_n| \theta)  \nonumber \\
    &= \sum_{b=1}^N  \underbrace{\left(\frac{1}{N}D_{KL}(q(\theta|\xi)|P(\theta)) -  \mathbb{E}_q \log P(D_n| \theta) \right)}_{\textrm{elbo}_n}
\end{align}
$$

where we denote the contribution from each individual datapoint as $\textrm{elbo}_n$. The expectations within this expression are relative to the surrogate distribution. In practice, we estimate these expectations using Monte-Carlo sampling. For a fixed $\xi$, we take samples from the surrogate distribution $q(\theta|\xi)$,

$$
\boldsymbol\theta_1,\ldots \boldsymbol\theta_k\ldots \boldsymbol\theta_K \sim q(\theta|\boldsymbol\xi),
$$

and approximate $\textrm{elbo}_n$ using a general expression

$$
\textrm{elbo}_n \approx \sum_{k=1}^K w_k\left[ \frac{1}{N} \frac{\log q(\theta_k|\xi)}{P(\theta_k)}  - \log P(D_n | \theta_k)\right],
$$
where $w_k$ are importance weights.

# TFP's built-in routines



`q_lp = surrogate_posterior.log_prob(q_samples)`


`divergence_fn` computes the vector quantities `q_lp`
$$
q_{lp} = \left( \log q_k(\theta_k) \right)_k,
$$
and `target_log_prob`
$$
t_{lp} = \left(\sum_n \log\Pr(y_n | \boldsymbol\theta_k)  + \log\Pr(\boldsymbol\theta_k) \right)_k.
$$

These quantities are then used to compute the vector valued `log_weights =  target_log_prob - q_lp`:

$$
lw = t_{lp} - q_{lp} = \left( \sum_n \log\Pr(y_n | \boldsymbol\theta_k)  + \log\frac{\Pr(\boldsymbol\theta_k)}{q(\boldsymbol\theta_k)} \right)_k
$$

The default `discrepancy_fn = tf.vi.kl_reverse` is

$$
f(x) = x \exp ^x
$$
so when applied component-wise,

$$
f(lw) = \left[  \left( \sum_n \log\Pr(y_n | \boldsymbol\theta_k)  + \log\frac{\Pr(\boldsymbol\theta_k)}{q(\boldsymbol\theta_k)} \right) \frac{\Pr(\boldsymbol\theta_k)}{q(\boldsymbol\theta_k)}  \prod_n \Pr(y_n | \boldsymbol\theta_k) \right]_k.
$$