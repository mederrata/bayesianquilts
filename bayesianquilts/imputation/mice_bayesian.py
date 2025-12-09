
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from bayesianquilts.imputation.mice import MICELogistic, ordinal_one_hot_encode
from bayesianquilts.models.logistic import BayesianMultinomialRegression
from bayesianquilts.util import training_loop
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior

class MICEBayesian(MICELogistic):
    """
    MICE using Bayesian Multinomial Regression with Horseshoe Priors.
    """
    def __init__(self, n_imputations=5, max_iter=5, random_state=42, n_predictors=25, 
                 global_shrinkage=0.01, batch_size=100, epochs=50):
        super().__init__(n_imputations, max_iter, random_state, n_predictors)
        self.global_shrinkage = global_shrinkage
        self.batch_size = batch_size
        self.epochs = epochs

    def fit_transform(self, X_df):
        # Override to use Bayesian model
        X = X_df.values
        N, M = X.shape
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        
        valid_vals = X[~missing_mask]
        max_val = int(valid_vals.max()) if len(valid_vals) > 0 else 1
        
        # Initialization (same as base)
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
            print(f"MICE Iteration {iteration+1}/{self.max_iter}")
            visit_order = np.arange(M)
            
            for j in visit_order:
                is_missing = missing_mask[:, j]
                if not np.any(is_missing):
                    continue
                
                obs_idx = np.where(~is_missing)[0]
                mis_idx = np.where(is_missing)[0]
                
                y_train = X_filled[obs_idx, j]
                
                # Predictor selection
                if M - 1 > self.n_predictors:
                    local_rng = np.random.RandomState(j)
                    candidates = np.delete(np.arange(M), j)
                    predictor_indices = local_rng.choice(candidates, size=self.n_predictors, replace=False)
                else:
                    predictor_indices = np.delete(np.arange(M), j)
                
                X_minus_j = X_filled[:, predictor_indices]
                
                y_train_unique = np.unique(y_train)
                if len(y_train_unique) < 2:
                    only_val = y_train_unique[0]
                    X_filled[mis_idx, j] = only_val
                    if iteration == self.max_iter - 1:
                        self.fitted_models_[j] = {'type': 'constant', 'value': int(only_val)}
                        for idx_global in mis_idx:
                            final_pmfs[(idx_global, j)] = {int(only_val): 1.0}
                    continue
                
                # Thermometer encoding
                X_features = ordinal_one_hot_encode(X_minus_j, max_val)
                input_dim = X_features.shape[1]
                
                # We need to map y_train to 0..K-1 for the model, as it expects categorical indices
                # Assuming y values are 0..max_val, but locally we might have gaps or subset.
                # However, ordinal regression usually implies we predict the LEVEL.
                # BayesianMultinomialRegression predicts classes 0..num_classes-1.
                # If we have levels 0, 1, 2, 3, let's treat them as classes.
                # But need to ensure num_classes covers the range.
                num_classes = max_val + 1
                
                # Bayesian Model
                bq_model = BayesianMultinomialRegression(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    global_shrinkage=self.global_shrinkage
                )
                
                # Prepare data factory for BQ
                # X_train: X_features[obs_idx], y_train: y_train
                X_train_obs = X_features[obs_idx]
                y_train_obs = y_train.astype(np.int32)
                
                ds_size = len(obs_idx)
                
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
                # BQ fit returns (loss, params)
                hist = bq_model.fit(
                    data_factory,
                    batch_size=self.batch_size,
                    dataset_size=ds_size,
                    checkpoint_dir=None,
                    num_epochs=self.epochs
                )
                
                params = bq_model.params
                
                # Sample model parameters from surrogate for stochastic imputation
                surrogate = bq_model.surrogate_distribution_generator(params)
                seed = jax.random.PRNGKey(self.rng.randint(0, 2**30))
                model_params_sample = surrogate.sample(seed=seed)
                
                # Transform log_tau/log_lam if predict_probs needed them, but it only uses beta/intercept.
                # The surrogate samples 'beta' and 'intercept' directly as proper parameters.
                
                # Predict
                X_miss = X_features[mis_idx]
                probs = bq_model.predict_probs(model_params_sample, X_miss)
                
                # Sample
                cumsum = probs.cumsum(axis=1)
                rand_vals = self.rng.rand(len(mis_idx), 1)
                choices_idx = (cumsum < rand_vals).sum(axis=1)
                # choices_idx are 0..num_classes-1. correspond directly to values if values are 0..max_val
                imputed_values = choices_idx 
                
                X_filled[mis_idx, j] = imputed_values
                
                if iteration == self.max_iter - 1:
                    self.fitted_models_[j] = {
                        'type': 'bayesian',
                        'params': params,
                        'predictor_indices': predictor_indices,
                        'max_val': max_val,
                        'num_classes': num_classes,
                        'global_shrinkage': self.global_shrinkage
                    }
                    classes = np.arange(num_classes)
                    for i_local, idx_global in enumerate(mis_idx):
                        pmf = {int(c): float(p) for c, p in zip(classes, probs[i_local])}
                        final_pmfs[(idx_global, j)] = pmf
                        
        return X_filled, final_pmfs

    def predict_pmfs(self, X_df, n_burnin=5):
        # Implementation similar to MICELogistic but using BQ models
        # For simplicity, we can rely on MICELogistic logic if we adapt the model prediction call
        # But prediction in MICELogistic assumes sklearn model.predict_proba.
        # Here we have stored 'params' and need to reinstantiate BQ model or just use a helper.
        
        # We can implement a lightweight predictor here.
        columns = X_df.columns
        X = X_df.values
        N, M = X.shape
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        
        max_val = 0
        for info in self.fitted_models_.values():
            if 'max_val' in info: max_val = max(max_val, info['max_val'])
        
        # Init
        for j in range(M):
            col_data = X[:, j]
            mis_indices = np.where(np.isnan(col_data))[0]
            if len(mis_indices) > 0:
                X_filled[mis_indices, j] = self.rng.randint(0, max_val + 1, size=len(mis_indices))
        X_filled = X_filled.astype(int)
        
        final_pmfs = {}
        
        for iteration in range(n_burnin):
            for j in range(M):
                if j not in self.fitted_models_: continue
                info = self.fitted_models_[j]
                
                is_missing = missing_mask[:, j]
                mis_idx = np.where(is_missing)[0]
                if len(mis_idx) == 0: continue
                
                if info['type'] == 'constant':
                    X_filled[mis_idx, j] = info['value']
                    if iteration == n_burnin - 1:
                         for idx in mis_idx: final_pmfs[(idx, j)] = {int(info['value']): 1.0}
                    continue
                
                params = info['params']
                pred_idx = info['predictor_indices']
                
                X_pred = X_filled[:, pred_idx]
                X_feat = ordinal_one_hot_encode(X_pred, max_val)
                X_target = X_feat[mis_idx]
                
                # Re-instantiate model to access surrogate generator
                bq_model = BayesianMultinomialRegression(
                    input_dim=X_feat.shape[1],
                    num_classes=info['num_classes'],
                    global_shrinkage=info['global_shrinkage']
                )
                
                # Sample parameters
                surrogate = bq_model.surrogate_distribution_generator(params)
                seed = jax.random.PRNGKey(self.rng.randint(0, 2**30))
                model_params_sample = surrogate.sample(seed=seed)
                
                # Predict
                probs = bq_model.predict_probs(model_params_sample, X_target)
                
                # Sample
                cumsum = probs.cumsum(axis=1)
                rand_vals = self.rng.rand(len(mis_idx), 1)
                choices = (cumsum < rand_vals).sum(axis=1)
                X_filled[mis_idx, j] = choices
                
                if iteration == n_burnin - 1:
                    classes = np.arange(info['num_classes'])
                    for i_local, idx_global in enumerate(mis_idx):
                        final_pmfs[(idx_global, j)] = {int(c): float(p) for c, p in zip(classes, probs[i_local])}
                        
        return final_pmfs
