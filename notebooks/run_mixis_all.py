#!/usr/bin/env python3
"""Run MixIS evaluation across all models for Table 1."""

import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from bayesianquilts.metrics.ais import LogisticRegressionLikelihood, MixIS, LikelihoodFunction
from bayesianquilts.metrics import nppsis

jax.config.update("jax_enable_x64", True)


def run_mixis_batched(likelihood, params, data, n_sims=25, n_samples=1000,
                      total_pool=None, batch_obs=None):
    """Run MixIS evaluation with optional observation batching."""
    mixis = MixIS(likelihood)
    leaves = jax.tree_util.tree_leaves(params)
    pool_size = total_pool or leaves[0].shape[0]

    results = []
    for sim_idx in range(n_sims):
        sample_ndx = np.random.choice(pool_size, size=n_samples, replace=False)
        sub_params = jax.tree_util.tree_map(lambda p: p[sample_ndx], params)

        log_ell = likelihood.log_likelihood(data, sub_params)  # (S, N)
        _, identity_khat = nppsis.psislw(-log_ell)
        n_bad = int(np.sum(identity_khat >= 0.7))

        if batch_obs is not None:
            N = log_ell.shape[1]
            all_khat = []
            for start in range(0, N, batch_obs):
                end = min(start + batch_obs, N)
                batch_data = jax.tree_util.tree_map(
                    lambda x: x[start:end] if x.ndim >= 1 and x.shape[0] == N else x,
                    data
                )
                batch_log_ell = log_ell[:, start:end]
                res = mixis(
                    max_iter=1, params=sub_params, theta=None,
                    data=batch_data, log_ell=batch_log_ell,
                    log_ell_original=log_ell,
                    rng_key=jax.random.PRNGKey(sim_idx),
                )
                all_khat.append(np.array(res['khat']))
            mixis_khat = np.concatenate(all_khat)
        else:
            res = mixis(
                max_iter=1, params=sub_params, theta=None,
                data=data, log_ell=log_ell, log_ell_original=log_ell,
                rng_key=jax.random.PRNGKey(sim_idx),
            )
            mixis_khat = np.array(res['khat'])

        n_fail = int(np.sum(mixis_khat >= 0.7))
        results.append({'n_bad': n_bad, 'n_fail': n_fail})
        print(f'  Sim {sim_idx+1}/{n_sims}: n_bad={n_bad}, MixIS failures={n_fail}')

    df = pd.DataFrame(results)
    mean_fail = df.n_fail.mean()
    std_fail = df.n_fail.std()
    print(f'  => MixIS failures: {mean_fail:.1f} +/- {std_fail:.1f}')
    return mean_fail, std_fail


class ReLUNetLikelihood(LikelihoodFunction):
    """Likelihood for shallow ReLU net."""

    def log_likelihood(self, data, params):
        X = jnp.asarray(data['X'], dtype=jnp.float64)
        y = jnp.asarray(data['y'], dtype=jnp.float64)
        w0, b0 = params['w_0'], params['b_0']
        w1, b1 = params['w_1'], params['b_1']
        hidden = jax.nn.relu(jnp.einsum('nf,sfh->snh', X, w0) + b0[:, None, :])
        mu = jnp.einsum('snh,sho->sno', hidden, w1)[..., 0] + b1
        sigma = jax.nn.sigmoid(mu)
        return y[None, :] * jnp.log(sigma + 1e-10) + (1 - y[None, :]) * jnp.log(1 - sigma + 1e-10)

    def log_likelihood_gradient(self, data, params):
        raise NotImplementedError
    def log_likelihood_hessian_diag(self, data, params):
        raise NotImplementedError


if __name__ == '__main__':
    base = Path.home() / 'workspace' / 'bayesianquilts'

    # ==================== OVARIAN LR ====================
    print("=" * 50)
    print("OVARIAN - Logistic Regression")
    print("=" * 50)

    X_ = pd.read_csv(base / 'python' / 'bayesianquilts' / 'data' / 'overianx.csv', header=None)
    y_ = pd.read_table(base / 'python' / 'bayesianquilts' / 'data' / 'overiany.csv', header=None)
    X_scaled = (X_ - X_.mean()) / X_.std()
    X_scaled = X_scaled.fillna(0).to_numpy()

    with open(base / 'notebooks' / 'ovarian' / '.cache' / 'ovarian_lr_stan_samples.pkl', 'rb') as f:
        lr_cached = pickle.load(f)

    lr_params = {
        'beta': jnp.array(lr_cached['beta'], dtype=jnp.float64),
        'intercept': jnp.array(lr_cached['beta0'][:, None], dtype=jnp.float64),
    }
    lr_data = {'X': jnp.array(X_scaled, dtype=jnp.float64),
               'y': jnp.array(y_.to_numpy()[:, 0], dtype=jnp.float64)}

    lr_mean, lr_std = run_mixis_batched(
        LogisticRegressionLikelihood(dtype=jnp.float64),
        lr_params, lr_data, n_sims=25, n_samples=1000
    )

    # ==================== OVARIAN NN ====================
    print("\n" + "=" * 50)
    print("OVARIAN - ReLU Neural Network")
    print("=" * 50)

    relu_raw = np.load(base / 'notebooks' / 'ovarian' / 'ovarian_relunet_params.npy', allow_pickle=True).item()
    with open(base / 'notebooks' / 'ovarian' / 'ovarian_relu_data.json') as f:
        relu_data_raw = json.load(f)

    nn_params = {k: jnp.array(relu_raw[k], dtype=jnp.float64) for k in ['w_0', 'b_0', 'w_1', 'b_1']}
    nn_data = {'X': jnp.array(relu_data_raw['X'], dtype=jnp.float64),
               'y': jnp.array(relu_data_raw['y'], dtype=jnp.float64)}

    nn_mean, nn_std = run_mixis_batched(
        ReLUNetLikelihood(), nn_params, nn_data,
        n_sims=25, n_samples=1000
    )

    # ==================== ROACH POISSON ====================
    print("\n" + "=" * 50)
    print("ROACH - Poisson Regression")
    print("=" * 50)

    from bayesianquilts.predictors.regression.poisson import PoissonRegressionLikelihood

    roach_df = pd.read_csv(base / 'python' / 'bayesianquilts' / 'data' / 'roachdata.csv')
    if 'Unnamed: 0' in roach_df.columns:
        roach_df = roach_df.drop(columns=['Unnamed: 0'])

    # Poisson features: roach1, treatment, senior + 3 interactions = 6
    X_df = roach_df[['roach1', 'treatment', 'senior']].copy()
    X_df['roach1_treatment'] = X_df['roach1'] * X_df['treatment']
    X_df['roach1_senior'] = X_df['roach1'] * X_df['senior']
    X_df['treatment_senior'] = X_df['treatment'] * X_df['senior']
    X_poisson = X_df.values.astype(np.float64)
    y_roach = roach_df['y'].values.astype(np.float64)
    offset_roach = np.log(roach_df['exposure2'].values).astype(np.float64)

    with open(base / 'notebooks' / 'roach' / '.cache' / 'poisson_mcmc_samples.pkl', 'rb') as f:
        pr_cached = pickle.load(f)

    pr_beta = pr_cached['beta'].reshape(-1, pr_cached['beta'].shape[-1])
    pr_intercept = pr_cached['intercept'].reshape(-1, pr_cached['intercept'].shape[-1])

    pr_params = {
        'beta': jnp.array(pr_beta, dtype=jnp.float64),
        'intercept': jnp.array(pr_intercept, dtype=jnp.float64),
    }
    pr_data = {
        'X': jnp.array(X_poisson, dtype=jnp.float64),
        'y': jnp.array(y_roach, dtype=jnp.float64),
        'offset': jnp.array(offset_roach, dtype=jnp.float64),
    }

    pr_mean, pr_std = run_mixis_batched(
        PoissonRegressionLikelihood(dtype=jnp.float64),
        pr_params, pr_data, n_sims=25, n_samples=1000,
        batch_obs=32
    )

    # ==================== ROACH NEGBIN ====================
    print("\n" + "=" * 50)
    print("ROACH - Negative Binomial Regression")
    print("=" * 50)

    from bayesianquilts.predictors.regression.negbin import NegativeBinomialRegressionLikelihood

    # NegBin features: roach1, treatment, senior, exposure2 = 4
    X_negbin = roach_df[['roach1', 'treatment', 'senior', 'exposure2']].values.astype(np.float64)

    with open(base / 'notebooks' / 'roach' / '.cache' / 'negbin_mcmc_samples.pkl', 'rb') as f:
        nb_cached = pickle.load(f)

    nb_beta = nb_cached['beta'].reshape(-1, nb_cached['beta'].shape[-1])
    nb_intercept = nb_cached['intercept'].reshape(-1, nb_cached['intercept'].shape[-1])
    nb_log_conc = nb_cached['log_concentration'].reshape(-1, nb_cached['log_concentration'].shape[-1])

    nb_params = {
        'beta': jnp.array(nb_beta, dtype=jnp.float64),
        'intercept': jnp.array(nb_intercept, dtype=jnp.float64),
        'log_concentration': jnp.array(nb_log_conc, dtype=jnp.float64),
    }
    nb_data = {
        'X': jnp.array(X_negbin, dtype=jnp.float64),
        'y': jnp.array(y_roach, dtype=jnp.float64),
    }

    nb_mean, nb_std = run_mixis_batched(
        NegativeBinomialRegressionLikelihood(dtype=jnp.float64),
        nb_params, nb_data, n_sims=25, n_samples=1000,
        batch_obs=32
    )

    # ==================== SUMMARY ====================
    print("\n" + "=" * 50)
    print("SUMMARY FOR TABLE 1")
    print("=" * 50)
    print(f"Ovarian LR:   ${lr_mean:.1f}\\pm{lr_std:.1f}$")
    print(f"Ovarian NN:   ${nn_mean:.1f}\\pm{nn_std:.1f}$")
    print(f"Roach PR:     ${pr_mean:.1f}\\pm{pr_std:.1f}$")
    print(f"Roach NBR:    ${nb_mean:.1f}\\pm{nb_std:.1f}$")
    print(f"\nLaTeX row:")
    print(f"& MixIS & ${lr_mean:.1f}\\pm{lr_std:.1f}$ & ${nn_mean:.1f}\\pm{nn_std:.1f}$ & ${pr_mean:.1f}\\pm{pr_std:.1f}$ & ${nb_mean:.1f}\\pm{nb_std:.1f}$ & --\\\\")
