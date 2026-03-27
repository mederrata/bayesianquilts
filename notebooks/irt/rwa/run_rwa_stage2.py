"""RWA Stage 2: MICE LOO fitting (loads saved baseline, runs MICE, saves)."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import polars as pl
from bayesianquilts.data.rwa import get_data, item_keys
from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

df, num_people = get_data(polars_out=True)
print(f"Loaded: {num_people} people, {len(item_keys)} items", flush=True)

imputation_df = df.select(item_keys).to_pandas()
imputation_df = imputation_df.replace(-1, float('nan'))
print(f"Missing values: {imputation_df.isna().sum().sum()}", flush=True)

pairwise_model = PairwiseOrdinalStackingModel(
    prior_scale=1.0, pathfinder_num_samples=100,
    pathfinder_maxiter=50, batch_size=256, verbose=True,
)
pairwise_model.fit(
    imputation_df, n_top_features=22, n_jobs=1,
)
pairwise_model.save('pairwise_stacking_model.yaml')
print('Pairwise stacking model saved', flush=True)
