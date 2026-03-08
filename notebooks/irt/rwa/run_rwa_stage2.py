"""RWA Stage 2: MICE LOO fitting (loads saved baseline, runs MICE, saves)."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import polars as pl
from bayesianquilts.data.rwa import get_data, item_keys
from bayesianquilts.imputation.mice_loo import MICEBayesianLOO

df, num_people = get_data(polars_out=True)
print(f"Loaded: {num_people} people, {len(item_keys)} items")

imputation_df = df.select(item_keys).to_pandas()
imputation_df = imputation_df.replace(-1, float('nan'))
print(f"Missing values per item:\n{imputation_df.isna().sum()}")

mice_loo = MICEBayesianLOO(
    prior_scale=1.0, pathfinder_num_samples=100,
    pathfinder_maxiter=50, batch_size=512, verbose=True,
)
mice_loo.fit_loo_models(
    imputation_df, n_top_features=22, n_jobs=1,
    fit_zero_predictors=True,
)
mice_loo.save('mice_loo_model.yaml')
print("MICE LOO saved")
