
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import arviz as az
from bayesianquilts.imputation.mice import MICELogistic, ordinal_one_hot_encode
from bayesianquilts.models.logistic import BayesianMultinomialRegression
from bayesianquilts.util import training_loop

class MICEBayesianLOO(MICELogistic):
    """
    MICE using Bayesian Multinomial Regression with LOO-based model weighting.
    For each variable to be imputed, we fit univariate models on all other variables,
    compute their LOO scores, and use these scores to weight the predictions.
    """
    def __init__(self, n_imputations=5, max_iter=5, random_state=42, 
                 global_shrinkage=0.01, batch_size=100, epochs=50, n_posterior_samples=100):
        # n_predictors is not used for selection, as we use all univariates
        super().__init__(n_imputations, max_iter, random_state, n_predictors=1)
        self.global_shrinkage = global_shrinkage
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_posterior_samples = n_posterior_samples

    def fit_transform(self, X_df):
        X = X_df.values
        N, M = X.shape
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        
        valid_vals = X[~missing_mask]
        max_val = int(valid_vals.max()) if len(valid_vals) > 0 else 1
        num_classes = max_val + 1
        
        # Initialization
        for j in range(M):
            col_data = X[:, j]
            obs_indices = np.where(~np.isnan(col_data))[0]
            mis_indices = np.where(np.isnan(col_data))[0]
            if len(mis_indices) > 0:
                if len(obs_indices) > 0:
                    fill_vals = self.rng.choice(col_data[obs_indices], size=len(mis_indices))
                    X_filled[mis_indices, j] = fill_vals
                else:
                    X_filled[mis_indices, j] = 0
        X_filled = X_filled.astype(int)
        
        final_pmfs = {}
        self.fitted_models_ = {}
        
        for iteration in range(self.max_iter):
            print(f"MICE LOO Iteration {iteration+1}/{self.max_iter}")
            visit_order = np.arange(M)
            
            for j in visit_order:
                is_missing = missing_mask[:, j]
                if not np.any(is_missing):
                    continue
                
                obs_idx = np.where(~is_missing)[0]
                mis_idx = np.where(is_missing)[0]
                
                y_train = X_filled[obs_idx, j]
                y_train_unique = np.unique(y_train)
                
                if len(y_train_unique) < 2:
                    only_val = y_train_unique[0]
                    X_filled[mis_idx, j] = only_val
                    if iteration == self.max_iter - 1:
                        # Store as a dummy model
                        self.fitted_models_[j] = {'type': 'constant', 'value': int(only_val)}
                        for idx_global in mis_idx:
                            final_pmfs[(idx_global, j)] = {int(only_val): 1.0}
                    continue

                # Prepare for averaging
                loo_scores = []
                model_predictions = [] # List of (model, params, X_feature_dim)
                
                other_vars = [k for k in range(M) if k != j]
                
                # Fit univariate models
                for k in other_vars:
                    # Predictor X_k
                    # We use filled values
                    X_k = X_filled[:, k:k+1]
                    
                    # Thermometer encoding for X_k
                    # X_k has values 0..max_val
                    X_features = ordinal_one_hot_encode(X_k, max_val)
                    input_dim = X_features.shape[1]
                    
                    # Model
                    bq_model = BayesianMultinomialRegression(
                        input_dim=input_dim,
                        num_classes=num_classes,
                        global_shrinkage=self.global_shrinkage
                    )
                    
                    X_train_obs = X_features[obs_idx]
                    y_train_obs = y_train.astype(np.int32)
                    ds_size = len(obs_idx)
                    
                    # Data factory
                    def data_factory(batch_size=None):
                        if batch_size is None: batch_size = self.batch_size
                        indices = np.arange(ds_size)
                        while True:
                            np.random.shuffle(indices)
                            for start in range(0, ds_size, batch_size):
                                idx = indices[start:start+batch_size]
                                yield {
                                    'X': X_train_obs[idx],
                                    'y': y_train_obs[idx]
                                }
                    
                    # Fit
                    bq_model.fit(
                        data_factory,
                        batch_size=self.batch_size,
                        dataset_size=ds_size,
                        checkpoint_dir=None,
                        num_epochs=self.epochs,
                        verbose=False
                    )
                    
                    params = bq_model.params
                    
                    # Sample posterior for LOO
                    surrogate = bq_model.surrogate_distribution_generator(params)
                    seed = jax.random.PRNGKey(self.rng.randint(0, 2**30))
                    posterior_samples = surrogate.sample(seed=seed, sample_shape=self.n_posterior_samples)
                    
                    # Calculate Log Likelihood (S, N_obs)
                    log_lik = bq_model.log_likelihood({
                        'X': X_train_obs, 
                        'y': y_train_obs
                    }, posterior_samples)
                    
                    # To numpy
                    log_lik_np = np.array(log_lik)
                    
                    # Compute LOO using ArviZ
                    # Create InferenceData
                    idata = az.from_dict(log_likelihood={'y': log_lik_np[None, ...]}) 
                    # Note: ArviZ expects (chains, draws, observations). 
                    # We have (draws, observations). Add chain dim 0.
                    
                    try:
                        loo_res = az.loo(idata, pointwise=False)
                        elpd_loo = loo_res.elpd_loo
                    except Exception as e:
                        print(f"Warning: LOO computation failed for var {j} predictor {k}: {e}")
                        elpd_loo = -np.inf

                    loo_scores.append(elpd_loo)
                    
                    # Predict on missing
                    X_miss = X_features[mis_idx]
                    probs = bq_model.predict_probs(posterior_samples, X_miss)
                    # probs: (S, N_miss, K)
                    # Average over posterior samples to get predictive probability
                    mean_probs = probs.mean(axis=0) # (N_miss, K)
                    
                    model_predictions.append(mean_probs)

                # Weighting
                loo_scores = np.array(loo_scores)
                # Handle -inf
                if np.all(np.isinf(loo_scores)):
                    weights = np.ones(len(loo_scores)) / len(loo_scores)
                else:
                    # Model averaging weights: w_k \propto exp(elpd_loo_k)
                    # Subtract max for stability
                    max_score = np.max(loo_scores)
                    weights = np.exp(loo_scores - max_score)
                    weights /= weights.sum()
                
                # Combine predictions
                # weighted sum of probabilities
                # model_predictions[k] is (N_miss, K)
                # weights[k] is scalar
                
                combined_probs = np.zeros_like(model_predictions[0])
                for w, preds in zip(weights, model_predictions):
                    combined_probs += w * preds
                    
                # Sample from combined probs
                cumsum = combined_probs.cumsum(axis=1)
                rand_vals = self.rng.rand(len(mis_idx), 1)
                choices_idx = (cumsum < rand_vals).sum(axis=1)
                X_filled[mis_idx, j] = choices_idx
                
                # Store PMFs if last iteration
                if iteration == self.max_iter - 1:
                    classes = np.arange(num_classes)
                    for i_local, idx_global in enumerate(mis_idx):
                        pmf = {int(c): float(p) for c, p in zip(classes, combined_probs[i_local])}
                        final_pmfs[(idx_global, j)] = pmf
                        
                    # Store info for prediction (optional, omitted for brevity as fit_transform is main goal)
                    # For full object support, we'd need to store all univariate models? 
                    # That would be huge. Maybe just store the weights and coefficients?
                    # The user prompt focused on "fitting", so fit_transform behavior is primary.
                    self.fitted_models_[j] = {
                        'type': 'ensemble_loo',
                        'weights': weights,
                        # 'sub_models': ... # If we need predict_pmfs later, we'd need to store them.
                    }

        return X_filled, final_pmfs
