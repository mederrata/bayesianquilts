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
    encoded = np.zeros((N, M * max_val), dtype=int)
    
    for v in range(1, max_val + 1):
        # Mask where value >= v
        mask = (data_matrix >= v)
        for col_idx in range(M):
            target_col = col_idx * max_val + (v - 1)
            encoded[:, target_col] = mask[:, col_idx].astype(int)
            
    return encoded

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
                - missing_probs (dict): Dictionary mapping (row_idx, col_idx) tuples to 
                  Probabilit Mass Functions {outcome_val: prob}.
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
        
        # Store PMFs for missing values: (row, col) -> {val: prob}
        final_pmfs = {} 
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
                         for i_local, idx_global in enumerate(mis_idx):
                             final_pmfs[(idx_global, j)] = {int(only_val): 1.0}
                             
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
                        multi_class='multinomial', 
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
                        for i_local, idx_global in enumerate(mis_idx):
                            pmf = {int(c): float(p) for c, p in zip(classes, probs[i_local])}
                            final_pmfs[(idx_global, j)] = pmf
                        
                        self.fitted_models_[j] = {
                            'type': 'logistic',
                            'model': model,
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
        
        # Determine global max for encoding from the training metadata or current data
        # Ideally we stored max_val or similar. 
        # In fit_transform we stored max_val in fitted_models_ entries.
        # Let's verify consistent max_val across models or pick the max.
        max_val = 0
        for info in self.fitted_models_.values():
            if 'max_val' in info:
                max_val = max(max_val, info['max_val'])
        
        if max_val == 0:
            # Fallback if not stored or constant models only
            valid_vals = X[~missing_mask]
            max_val = int(valid_vals.max()) if len(valid_vals) > 0 else 1

        # Initial fill
        for j in range(M):
            col_data = X[:, j]
            mis_indices = np.where(np.isnan(col_data))[0]
            if len(mis_indices) > 0:
                # Fill with random valid value (e.g. 0) or mean/mode if we had it.
                # Random integer 0..max_val is a safe bet for init
                X_filled[mis_indices, j] = self.rng.randint(0, max_val + 1, size=len(mis_indices))
        
        X_filled = X_filled.astype(int)
        
        # Gibbs Sampling Loop
        final_pmfs = {}
        
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
                    if iteration == n_burnin - 1:
                        for idx_global in mis_idx:
                            final_pmfs[(idx_global, j)] = {int(val): 1.0}
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
                        for i_local, idx_global in enumerate(mis_idx):
                            pmf = {int(c): float(p) for c, p in zip(classes, probs[i_local])}
                            final_pmfs[(idx_global, j)] = pmf
                            
                except Exception:
                    continue
                    
        return final_pmfs
