
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tensorflow_probability.substrates import jax as tfp

from bayesianquilts.model import BayesianModel

tfd = tfp.distributions
tfb = tfp.bijectors


class VariationalMMGradedResponseModel(BayesianModel):
    """
    Variational Graded Response Model with MM-based handling of missing data
    imputed via MICE.
    
    This model fits an IRT model using a surrogate likelihood that averages 
    over the missing data distribution $g$ provided by MICE.
    
    $Q(\Xi) = \mathbb{E}_q [\sum_{obs} \log P(y_{obs} | \Xi) + \sum_{miss} \sum_k g(y_{miss}=k) \log P(y_{miss}=k | \Xi) ] + \dots$
    
    Data input format is expected to be a unified set of records where each record
    contains a probabilty mass function (PMF) over the outcomes. 
    - For observed data: PMF is a one-hot vector (probability 1.0 for the observed value).
    - For missing data: PMF is the output from MICE.
    """
    
    def __init__(
        self,
        num_respondents: int,
        num_items: int,
        num_outcomes: int,
        latent_dim: int = 1,
        dtype=jnp.float32,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_respondents = num_respondents
        self.num_items = num_items
        self.num_outcomes = num_outcomes
        self.latent_dim = latent_dim
        
        # Define variable list for the model (theta=ability, alpha=discrimination, kappa=thresholds)
        self.var_list = ['theta', 'alpha', 'kappa']
        
        # Initialize distributions
        self.create_distributions()
        
    def create_distributions(self):
        """Create the prior and surrogate distributions."""
        
        # --- Priors ---
        # Theta (Ability): Standard Normal
        # Shape: (num_respondents, latent_dim)
        prior_theta = tfd.Sample(
            tfd.Normal(loc=jnp.zeros([], dtype=self.dtype), scale=1.0),
            sample_shape=[self.num_respondents, self.latent_dim]
        )
        
        # Alpha (Discrimination): Lognormal (must be positive)
        # Shape: (num_items, latent_dim)
        # We model log_alpha in the surrogate, but the prior is on alpha.
        # Actually easier to interpret params as unconstrained and transform them.
        # But bayesianquilts typically handles bijectors separately or uses TransformedDistribution.
        # Here we'll define the prior on the *unconstrained* space for simplicity in VI, 
        # or use the distributions directly if the surrogate handles it.
        # Let's use unconstrained parameters in the surrogate (Normal) and transform them.
        
        # But here we define the mathematical prior distribution objects.
        
        # Alpha: LogNormal(0, 1) -> log_alpha ~ Normal(0, 1)
        prior_log_alpha = tfd.Sample(
            tfd.Normal(loc=jnp.zeros([], dtype=self.dtype), scale=1.0),
            sample_shape=[self.num_items, self.latent_dim]
        )
        
        # Kappa (Thresholds): Ordered
        # Shape: (num_items, num_outcomes - 1)
        # We need num_outcomes - 1 cutpoints for num_outcomes categories.
        # Prior on thresholds is often Normal(0, 2) roughly. 
        # But they must be ordered. 
        # Strategy: Parametrise as (first_cutpoint, log_gaps...)
        # or use TFP Ordered bijector.
        
        # For the prior, let's assume a broad Normal on the cutpoints, but restricted to be ordered.
        # This is tricky in VI. Standard approach: use unconstrained parameterization
        # [c1, log(c2-c1), log(c3-c2)...]
        
        # Let's assume prior on the *unconstrained* thresholds is standard normal-ish.
        prior_kappa_unconstrained = tfd.Sample(
            tfd.Normal(loc=jnp.zeros([], dtype=self.dtype), scale=2.0),
            sample_shape=[self.num_items, self.num_outcomes - 1]
        )
        
        self.prior_distribution = tfd.JointDistributionNamed({
            'theta': prior_theta, # Actually we typically don't fit theta with VI for global params? 
                                  # Wait, in IRT, theta are local parameters (per person). 
                                  # If N is large, we usually want to marginalize them or fit them with VI.
                                  # We will fit them relative to the "global" variational approximation.
            'log_alpha': prior_log_alpha,
            'kappa_unconstrained': prior_kappa_unconstrained
        })
        
        # --- Surrogate (Variational Posterior) ---
        # Mean-field Normal for all parameters
        
        self.surrogate_distribution = tfd.JointDistributionNamedAutoBatched({
            'theta': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(0), (self.num_respondents, self.latent_dim), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(
                    jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(1), (self.num_respondents, self.latent_dim), dtype=self.dtype)
                )
            ),
            'log_alpha': tfd.Normal(
                loc=jax.nn.initializers.zeros(jax.random.PRNGKey(2), (self.num_items, self.latent_dim), dtype=self.dtype),
                scale=1e-2 + jax.nn.softplus(
                    jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(3), (self.num_items, self.latent_dim), dtype=self.dtype)
                )
            ),
            'kappa_unconstrained': tfd.Normal(
                loc=jnp.sort(jax.random.normal(jax.random.PRNGKey(4), (self.num_items, self.num_outcomes - 1), dtype=self.dtype)), 
                # Init sorted to be close to valid, though unconstrained params don't strictly need sorting, 
                # their *values* map to sorted. Actually if we use (c1, log_gap), init should be (0, 0...)
                scale=1e-2 + jax.nn.softplus(
                    jax.nn.initializers.uniform(scale=0.01)(jax.random.PRNGKey(5), (self.num_items, self.num_outcomes - 1), dtype=self.dtype)
                )
            )
        })
        
        # To make "surrogate_distribution" a learnable object in bayesianquilts (which uses flax/nnx or objax?),
        # looking at bayesianquilts/model.py, it expects `surrogate_distribution` to be an attribute.
        # But typically one needs a `surrogate_distribution_generator(self, params)` if params are passed explicitly.
        # However, the base classes in bayesianquilts seem to imply `_calibrate_advi` handles 
        # creating the surrogate from `initial_values` dict or similar.
        # 
        # In `_calibrate_minibatch_advi`:
        # `surrogate_generator=self.surrogate_distribution_generator`
        # So I must implement `surrogate_distribution_generator` and `surrogate_parameter_initializer`.

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        """Initialize parameters for the surrogate distribution."""
        if key is None:
            key = jax.random.PRNGKey(42)
        # Mean and Log-Std for each variable
        
        # Helper for initialization
        def init_mean_scale(shape, key):
            k1, k2 = jax.random.split(key)
            mean = jax.random.normal(k1, shape, dtype=self.dtype) * 0.1
            # Initialize scale small
            raw_scale = jnp.log(jnp.exp(0.1) - 1.0) + jax.random.normal(k2, shape, dtype=self.dtype) * 0.01
            return mean, raw_scale

        k1, k2, k3 = jax.random.split(key, 3)
        
        theta_loc, theta_scale = init_mean_scale((self.num_respondents, self.latent_dim), k1)
        alpha_loc, alpha_scale = init_mean_scale((self.num_items, self.latent_dim), k2)
        
        # Initialize thresholds to be ordered-ish in expectation
        # For unconstrained: (c1, log_gap, log_gap...)
        # Let's define the transformation carefully later.
        # For now just init random.
        kappa_loc, kappa_scale = init_mean_scale((self.num_items, self.num_outcomes - 1), k3)
        
        return {
            'theta_loc': theta_loc,
            'theta_raw_scale': theta_scale,
            'log_alpha_loc': alpha_loc,
            'log_alpha_raw_scale': alpha_scale,
            'kappa_unconstrained_loc': kappa_loc,
            'kappa_unconstrained_raw_scale': kappa_scale
        }

    def surrogate_distribution_generator(self, params):
        """Generate surrogate distribution from parameters."""
        
        return tfd.JointDistributionNamed({
            'theta': tfd.Independent(tfd.Normal(
                loc=params['theta_loc'],
                scale=jax.nn.softplus(params['theta_raw_scale']) + 1e-4
            ), reinterpreted_batch_ndims=2),
            
            'log_alpha': tfd.Independent(tfd.Normal(
                loc=params['log_alpha_loc'],
                scale=jax.nn.softplus(params['log_alpha_raw_scale']) + 1e-4
            ), reinterpreted_batch_ndims=2),
            
            'kappa_unconstrained': tfd.Independent(tfd.Normal(
                loc=params['kappa_unconstrained_loc'],
                scale=jax.nn.softplus(params['kappa_unconstrained_raw_scale']) + 1e-4
            ), reinterpreted_batch_ndims=2)
        })

    def transform_kappa(self, kappa_unconstrained):
        """
        Transform unconstrained kappa parameters into ordered thresholds.
        Mapping:
        k_1 = u_1
        k_2 = k_1 + exp(u_2)
        ...
        """
        # Using cumsum of softplus (or exp) of the gaps
        # First element is typically free, subsequent are gaps.
        # However, a simpler stable transform is often used. 
        # TFP has FillScaleTriL. 
        # Here we manually do cumsum of exp.
        
        first = kappa_unconstrained[..., 0:1]
        gaps = jnp.exp(kappa_unconstrained[..., 1:]) # Force positive gaps
        
        # Or softplus for stability
        # gaps = jax.nn.softplus(kappa_unconstrained[..., 1:])
        
        rest = jnp.cumsum(gaps, axis=-1)
        return jnp.concatenate([first, first + rest], axis=-1)

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        """
        Compute the unnormalized log probability (Joint Log-Likelihood).
        
        Args:
            data: Dictionary containing:
                  - 'respondent_id': (Batch,) int
                  - 'item_id': (Batch,) int
                  - 'pmf_weights': (Batch, num_outcomes) float. 
                                   For observed data, one-hot.
                                   For missing data, MICE probabilities.
        """
        theta = params['theta'] # (N_respondents, D)
        log_alpha = params['log_alpha'] # (N_items, D)
        kappa_unconstrained = params['kappa_unconstrained'] # (N_items, K-1)
        
        alpha = jnp.exp(log_alpha)
        kappa = self.transform_kappa(kappa_unconstrained)
        
        # Look up parameters for the batch
        # params may have sample dimension (S, N, D) or not (N, D).
        # We index into N, which is axis -2.
        batch_theta = jnp.take(theta, data['respondent_id'], axis=-2) # (..., B, D)
        batch_alpha = jnp.take(alpha, data['item_id'], axis=-2)   # (..., B, D)
        batch_kappa = jnp.take(kappa, data['item_id'], axis=-2)   # (..., B, K-1)
        
        # Calculate Graded Response Model Probabilities
        # P(Y >= k) = sigmoid(alpha * (theta - kappa_k))
        # GRM usually: a * (theta - b_k).
        # Note dimensions: theta is (B, D), alpha is (B, D). vector product.
        # Inner term: alpha * theta is (B, D) * (B, D) -> sum -> (B,).
        # Multi-dimensional IRT: dot(alpha, theta).
        
        interaction = jnp.sum(batch_theta * batch_alpha, axis=-1, keepdims=True) # (B, 1)
        
        # We need logits for P(Y >= k).
        # P(Y >= k) = sigmoid(interaction - batch_kappa_k)
        # Note: batch_alpha is effectively a scaling factor on theta. 
        # The threshold kappa is subtracted.
        # Often it is D * (theta - b). Here we use D*theta - d.
        # Let's stick to D*theta - kappa.
        
        logits_cumulative = interaction - batch_kappa # (B, K-1)
        
        # Cumulative probabilities P(Y >= k)
        probs_cumulative = jax.nn.sigmoid(logits_cumulative)
        
        # Pad with 1.0 (P(Y>=0)) and 0.0 (P(Y>=K))
        # Shape becomes (B, K+1)
        # P(Y >= 0) = 1
        # P(Y >= 1) = probs_cumulative[:, 0]
        # ...
        # P(Y >= K) = 0
        
        # P(Y >= 0) = 1
        # P(Y >= 1) = probs_cumulative[:, 0]
        # ...
        # P(Y >= K) = 0
        
        target_shape = list(logits_cumulative.shape)
        target_shape[-1] = 1
        ones = jnp.ones(target_shape, dtype=self.dtype)
        zeros = jnp.zeros(target_shape, dtype=self.dtype)
        
        probs_ge = jnp.concatenate([ones, probs_cumulative, zeros], axis=-1) # (..., B, K+1)
        
        # Probability of category k: P(Y=k) = P(Y>=k) - P(Y>=k+1)
        probs_category = probs_ge[..., :-1] - probs_ge[..., 1:] # (..., B, K)
        
        # Avoid log(0)
        probs_category = jnp.clip(probs_category, 1e-10, 1.0)
        log_probs = jnp.log(probs_category) # (B, K)
        
        # Expected Log Likelihood
        # Weight by the input PMF
        # Input 'pmf_weights' should be (B, K)
        pmf = data['pmf_weights']
        
        # Sum over categories
        expected_ll_per_datum = jnp.sum(pmf * log_probs, axis=1) # (B,)
        
        # Total Log Likelihood
        log_likelihood = jnp.sum(expected_ll_per_datum)
        
        # Priors
        # We can penalize using the priors.
        # Since this is "unnormalized_log_prob", we need log_prior + log_likelihood.
        # However, because of minibatching, we upscale the LL, but the prior is computed once per epoch usually.
        # bayesianquilts handles scaling via `prior_weight` usually if we put priors in the loss.
        # But here we are passed 'prior_weight' which usually scales the likelihoood relative to prior? 
        # No, commonly `prior_weight` is < 1 (1/num_batches) to scale DOWN the prior.
        # Let's assume standard bayesianquilts practice: 
        # sum(log_prior) * prior_weight + log_likelihood
        
        log_prior = (
            jnp.sum(tfd.Normal(0., 1.).log_prob(params['theta'])) + 
            jnp.sum(tfd.Normal(0., 1.).log_prob(params['log_alpha'])) +
            jnp.sum(tfd.Normal(0., 2.).log_prob(params['kappa_unconstrained']))
        )
        
        return log_likelihood + log_prior * prior_weight

    def predictive_distribution(self, data, **params):
        """Compute predictive distribution for checking."""
        theta = params['theta']
        log_alpha = params['log_alpha']
        kappa_unconstrained = params['kappa_unconstrained']
        
        alpha = jnp.exp(log_alpha)
        kappa = self.transform_kappa(kappa_unconstrained)
        
        batch_theta = jnp.take(theta, data['respondent_id'], axis=-2)
        batch_alpha = jnp.take(alpha, data['item_id'], axis=-2)
        batch_kappa = jnp.take(kappa, data['item_id'], axis=-2)
        
        interaction = jnp.sum(batch_theta * batch_alpha, axis=-1, keepdims=True)
        logits_cumulative = interaction - batch_kappa
        probs_cumulative = jax.nn.sigmoid(logits_cumulative)
        
        target_shape = list(logits_cumulative.shape)
        target_shape[-1] = 1
        ones = jnp.ones(target_shape, dtype=self.dtype)
        zeros = jnp.zeros(target_shape, dtype=self.dtype)
        
        probs_ge = jnp.concatenate([ones, probs_cumulative, zeros], axis=-1)
        probs_category = probs_ge[..., :-1] - probs_ge[..., 1:]
        
        return {
            "probs": probs_category,
            "prediction": jnp.argmax(probs_category, axis=-1)
        }
        
    # --- Helper methods for data loading ---
    
    @staticmethod
    def prepare_data(df, pmf_lookup, respondent_col='Name', item_col='Item', outcome_col='Outcome', num_outcomes=4):
        """
        Static helper to prepare data for the model from a long-format DataFrame and MICE PMF lookup.
        
        Args:
            df: DataFrame in long format (must contain respondent_col, item_col, outcome_col for observed data).
                Missing data entries in df are ignored; they are filled from pmf_lookup.
            pmf_lookup: Dictionary or path to JSON containing MICE PMFs.
                        Keys should be f"{respondent}|{item}".
                        Values are {outcome_val: prob}.
            respondent_col: Name of column with respondent identifiers.
            item_col: Name of column with item identifiers.
            outcome_col: Name of column with observed outcomes (integers).
            num_outcomes: Total number of ordinal outcome categories.
            
        Returns:
            Dictionary suitable for model training, including mapped IDs and PMF weights.
        """
        
        # Load PMFs if path
        if isinstance(pmf_lookup, (str, Path)):
            with open(pmf_lookup, 'r') as f:
                pmf_lookup = json.load(f)
            
        # Create Mappings
        unique_respondents = sorted(df[respondent_col].unique())
        unique_items = sorted(df[item_col].unique())
        
        respondent_to_id = {n: i for i, n in enumerate(unique_respondents)}
        item_to_id = {n: i for i, n in enumerate(unique_items)}
        
        records_respondent = []
        records_item = []
        records_pmf = []
        
        processed_keys = set()
        
        # 1. Observed Data
        # Filter for observed outcomes
        df_observed = df.dropna(subset=[outcome_col])
        
        for _, row in df_observed.iterrows():
            outcome = int(row[outcome_col])
            r_name = row[respondent_col]
            i_name = row[item_col]
            
            if r_name not in respondent_to_id or i_name not in item_to_id:
                continue 
            
            rid = respondent_to_id[r_name]
            iid = item_to_id[i_name]
            
            # One-Hot PMF
            pmf = np.zeros(num_outcomes)
            if 0 <= outcome < num_outcomes:
                pmf[outcome] = 1.0
            else:
                logging.warning(f"Outcome {outcome} out of bounds")
                continue
                
            records_respondent.append(rid)
            records_item.append(iid)
            records_pmf.append(pmf)
            processed_keys.add(f"{r_name}|{i_name}")
        
        # 2. Missing Data from PMFs
        for key, probs in pmf_lookup.items():
            # Expected key format "Respondent|Item"
            # This is tightly coupled to how MICE saved it.
            if key in processed_keys:
                continue 
            
            parts = key.split('|')
            if len(parts) != 2: continue
            r_name, i_name = parts[0], parts[1]
            
            if r_name in respondent_to_id and i_name in item_to_id:
                rid = respondent_to_id[r_name]
                iid = item_to_id[i_name]
                
                pmf = np.zeros(num_outcomes)
                for val_str, prob in probs.items():
                    val = int(val_str)
                    if 0 <= val < num_outcomes:
                        pmf[val] = prob
                        
                records_respondent.append(rid)
                records_item.append(iid)
                records_pmf.append(pmf)
                
        return {
            'respondent_id': np.array(records_respondent, dtype=np.int32),
            'item_id': np.array(records_item, dtype=np.int32),
            'pmf_weights': np.array(records_pmf, dtype=np.float32),
            'metadata': {
                'respondent_to_id': respondent_to_id,
                'item_to_id': item_to_id
            }
        }

