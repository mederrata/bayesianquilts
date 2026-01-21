import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def ordinal_one_hot_encode(data_matrix, max_val):
    """
    Encodes integer data matrix into ordinal one-hot (thermometer) format.
    
    Args:
        data_matrix: (N, M) matrix of integers 0..max_val
        max_val: Maximum possible value (determines vector length per feature)
        
    Returns:
        Encoded matrix of shape (N, M * max_val)
    """
    N, M = data_matrix.shape
    encoded = np.zeros((N, M * max_val), dtype=np.int8)
    
    for v in range(1, max_val + 1):
        # Mask where value >= v
        mask = (data_matrix >= v)
        for col_idx in range(M):
            target_col = col_idx * max_val + (v - 1)
            encoded[:, target_col] = mask[:, col_idx].astype(int)
            
    return encoded
    
class LightweightLogisticModel:
    """Minimal storage for logistic regression parameters."""
    def __init__(self, sklearn_model):
        self.coef_ = sklearn_model.coef_.copy()
        self.intercept_ = sklearn_model.intercept_.copy()
        self.classes_ = sklearn_model.classes_.copy()
        
    def predict_proba(self, X):
        # Implementation of logistic probability prediction
        # z = X @ coef.T + intercept
        z = np.dot(X, self.coef_.T) + self.intercept_
        
        # Softmax
        # For binary/multinomial
        if z.shape[1] == 1:
            # Binary case (sklearn coef is (1, n_features))
            # Output proba is (N, 2)
            # sigmoid
            p1 = 1.0 / (1.0 + np.exp(-z))
            p0 = 1.0 - p1
            return np.hstack([p0, p1])
        else:
            # Multinomial
            # Stable softmax
            z_max = np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z - z_max)
            sum_exp = np.sum(exp_z, axis=1, keepdims=True)
            return exp_z / sum_exp
            return exp_z / sum_exp
            
class MICELogistic:
    """
    Multiple Imputation by Chained Equations (MICE) using Ordinal Logistic Regression.
    
    This implementation focuses on providing Probability Mass Functions (PMFs)
    for missing values, which can be useful for integrating over missing data uncertainty.
    """
    def __init__(self, n_imputations=10, max_iter=5, random_state=42, n_predictors=25):
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_predictors = n_predictors
        
    def fit_transform(self, X_df):
        """
        Run MICE imputation on a DataFrame.
        
        Args:
            X_df (pd.DataFrame): DataFrame with missing values (NaNs).
            
        Returns:
            tuple:
                - imputed_data (np.ndarray): One complete sample variable of imputed data.
            tuple:
                - imputed_data (np.ndarray): One complete sample variable of imputed data.
                - missing_probs (np.ndarray): Array of shape (N, M, max_val+1) containing 
                  PMFs for missing values. Entry [n, m, v] is P(X_nm = v).
        """
        X = X_df.values
        N, M = X.shape
        
        # Identify missingness
        missing_mask = np.isnan(X)
        
        # 1. Initialization: Fill NaNs with random observed values from same column
        X_filled = X.copy()
        
        # Determine global max value for encoding (assuming ordinal integer levels)
        valid_vals = X[~missing_mask]
        max_val = int(valid_vals.max()) if len(valid_vals) > 0 else 1
        
        for j in range(M):
            col_data = X[:, j]
            obs_indices = np.where(~np.isnan(col_data))[0]
            mis_indices = np.where(np.isnan(col_data))[0]
            
            if len(mis_indices) > 0:
                # Sample from observed
                if len(obs_indices) > 0:
                    fill_vals = self.rng.choice(col_data[obs_indices], size=len(mis_indices))
                    X_filled[mis_indices, j] = fill_vals
                else:
                    # If column is empty, fill with 0 or skip? 
                    # Assuming data prep handles empty columns.
                    X_filled[mis_indices, j] = 0
                
        X_filled = X_filled.astype(int)
        
        # Store PMFs for missing values: (N, M, max_val+1)
        # Using float32 to save memory
        final_pmfs = np.zeros((N, M, max_val + 1), dtype=np.float32)
        self.fitted_models_ = {} # col_idx -> {model, predictors, classes, etc}
        
        # 2. MICE Loop
        for iteration in range(self.max_iter):
            # Order of visiting variables
            visit_order = np.arange(M)
            
            for j in visit_order:
                # Identify observed/missing for THIS variable in ORIGINAL data
                is_missing = missing_mask[:, j]
                if not np.any(is_missing):
                    continue
                
                obs_idx = np.where(~is_missing)[0]
                mis_idx = np.where(is_missing)[0]
                
                # Target
                y_train = X_filled[obs_idx, j]
                
                # Predictors: Select subset to speed up
                if M - 1 > self.n_predictors:
                    # Deterministic selection based on column index j to keep it stable per column
                    local_rng = np.random.RandomState(j)
                    candidates = np.delete(np.arange(M), j)
                    predictor_indices = local_rng.choice(candidates, size=self.n_predictors, replace=False)
                    X_minus_j = X_filled[:, predictor_indices]
                else:
                    predictor_indices = np.delete(np.arange(M), j)
                    X_minus_j = X_filled[:, predictor_indices]
                
                y_train_unique = np.unique(y_train)
                if len(y_train_unique) < 2:
                    # Only one class observed. Impute deterministically.
                    only_val = y_train_unique[0]
                    imputed_values = np.full(len(mis_idx), only_val)
                    
                    if iteration == self.max_iter - 1:
                         # Deterministic (prob 1.0)
                         # We can broadcast assignment if mis_idx is large?
                         # final_pmfs[mis_idx, j, int(only_val)] = 1.0
                         final_pmfs[mis_idx, j, int(only_val)] = 1.0
                             
                         # Store "trivial" model
                         self.fitted_models_[j] = {
                             'type': 'constant',
                             'value': int(only_val)
                         }
                    
                    X_filled[mis_idx, j] = imputed_values
                    continue

                # Encode features
                X_features = ordinal_one_hot_encode(X_minus_j, max_val)
                
                # Train Model
                try:
                    model = LogisticRegression(
                        solver='lbfgs', 
                        max_iter=200, 
                        C=1.0,
                        random_state=self.random_state
                    )
                    
                    model.fit(X_features[obs_idx], y_train)
                    
                    # Predict Probabilities for missing
                    probs = model.predict_proba(X_features[mis_idx])
                    classes = model.classes_
                    
                    # Impute by sampling
                    cumsum = probs.cumsum(axis=1)
                    rand_vals = self.rng.rand(len(mis_idx), 1)
                    choices_idx = (cumsum < rand_vals).sum(axis=1)
                    imputed_values = classes[choices_idx]
                    
                    # Update matrix
                    X_filled[mis_idx, j] = imputed_values
                    
                    # If this is the last iteration, save probs and model
                    if iteration == self.max_iter - 1:
                        # Vectorized assignment of PMFs
                        # classes are the column indices in the last dimension
                        # probs is (len(mis_idx), len(classes))
                        
                        # We need to map class values to indices in final_pmfs (last dim)
                        # classes might not be 0..K contiguous if some levels missing?
                        # But max_val implies 0..max_val range.
                        # Assuming classes are integers fitting in max_val.
                        
                        for k, class_val in enumerate(classes):
                            # Assign column k of probs to ...
                            idx_cls = int(class_val)
                            final_pmfs[mis_idx, j, idx_cls] = probs[:, k].astype(np.float32)
                        
                        self.fitted_models_[j] = {
                            'type': 'logistic',
                            'model': LightweightLogisticModel(model),
                            'predictor_indices': predictor_indices,
                            'max_val': max_val
                        }
                            
                except Exception:
                    # Fallback on failure: keep previous imputed values
                    pass

        return X_filled, final_pmfs

    def predict_pmfs(self, X_df, n_burnin=5):
        """
        Generate Probability Mass Functions for missing values using fitted models.
        
        Args:
            X_df: DataFrame with missing values.
            n_burnin: Number of Gibbs sampling iterations to stabilize imputations 
                      before capturing the final PMF.
                      
        Returns:
            dict: Mapping {(row_idx, col_idx): {val: prob, ...}}
        """
        if not hasattr(self, 'fitted_models_'):
            raise RuntimeError("Model must be fitted before prediction.")
            
        columns = X_df.columns
        X = X_df.values
        N, M = X.shape
        missing_mask = np.isnan(X)
        
        # Initialization
        X_filled = X.copy()
        
        # Determine global max for encoding
        max_val = 0
        for info in self.fitted_models_.values():
            if 'max_val' in info:
                max_val = max(max_val, info['max_val'])
        
        if max_val == 0:
            valid_vals = X[~missing_mask]
            max_val = int(valid_vals.max()) if len(valid_vals) > 0 else 1

        # Use float32 array for output
        final_pmfs = np.zeros((N, M, max_val + 1), dtype=np.float32)
        
        for iteration in range(n_burnin):
            visit_order = np.arange(M)
            
            for j in visit_order:
                if j not in self.fitted_models_:
                    continue
                    
                info = self.fitted_models_[j]
                
                # Identify missing for THIS variable
                is_missing = missing_mask[:, j]
                mis_idx = np.where(is_missing)[0]
                
                if len(mis_idx) == 0:
                    continue
                
                if info['type'] == 'constant':
                    val = info['value']
                    X_filled[mis_idx, j] = val
                    X_filled[mis_idx, j] = val
                    if iteration == n_burnin - 1:
                        final_pmfs[mis_idx, j, int(val)] = 1.0
                    continue
                
                # Logistic prediction
                model = info['model']
                predictor_indices = info['predictor_indices']
                
                X_predictors = X_filled[:, predictor_indices]
                X_features = ordinal_one_hot_encode(X_predictors, max_val)
                
                try:
                    # Predict probs
                    probs = model.predict_proba(X_features[mis_idx])
                    classes = model.classes_
                    
                    # Sample new values
                    cumsum = probs.cumsum(axis=1)
                    rand_vals = self.rng.rand(len(mis_idx), 1)
                    choices_idx = (cumsum < rand_vals).sum(axis=1)
                    imputed_values = classes[choices_idx]
                    
                    X_filled[mis_idx, j] = imputed_values
                    
                    if iteration == n_burnin - 1:
                        for k, class_val in enumerate(classes):
                             idx_cls = int(class_val)
                             final_pmfs[mis_idx, j, idx_cls] = probs[:, k].astype(np.float32)
                            
                except Exception:
                    continue
                    
        return final_pmfs
