"""RWA Stage 3: Mixed imputation + imputed GRM + plots."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_helpers import (plot_loss_comparison, plot_forest_discriminations,
                          plot_ability_scatter, plot_ability_distributions,
                          plot_thresholds, plot_individual_abilities,
                          plot_imputation_weights_pcolormesh)

from bayesianquilts.data.rwa import get_data, item_keys
from bayesianquilts.irt.grm import GRModel
from bayesianquilts.imputation.mice_loo import MICEBayesianLOO
from bayesianquilts.imputation.mixed import IrtMixedImputationModel

response_cardinality = 9

df, num_people = get_data(polars_out=True)
SUBSAMPLE_N = num_people
sub_df = df
print(f"Loaded: {num_people} people, {len(item_keys)} items, K={response_cardinality}")

def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float32)
        data[col] = arr
    data['person'] = np.arange(len(dataframe), dtype=np.float32)
    return data

batch = make_data_dict(sub_df)
BATCH_SIZE = 256
steps_per_epoch = int(np.ceil(SUBSAMPLE_N / BATCH_SIZE))

def data_factory():
    indices = np.arange(SUBSAMPLE_N)
    np.random.shuffle(indices)
    for start in range(0, SUBSAMPLE_N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, SUBSAMPLE_N)
        idx_batch = indices[start:end]
        yield {k: v[idx_batch] for k, v in batch.items()}

# Load baseline model
print("Loading baseline GRM...")
model_baseline = GRModel(
    item_keys=item_keys, num_people=SUBSAMPLE_N, dim=1,
    response_cardinality=response_cardinality,
    dtype=jnp.float32,
        parameterization="log_scale",
)
# Initialize then load saved params
import h5py
with h5py.File('grm_baseline/params.h5', 'r') as f:
    loaded_params = {}
    for key in f.keys():
        loaded_params[key] = jnp.array(f[key][:])
model_baseline.params = loaded_params

# Load snapshot params from epoch 50
import pickle
with open('snapshot_params.pkl', 'rb') as f:
    snapshot_params = pickle.load(f)

def calibrate_manually(model, n_samples=32, seed=42):
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
        model.calibrated_expectations = expectations
        model.surrogate_sample = samples
    except KeyError as e:
        print(f"  Warning: surrogate sampling failed ({e}), using point estimates")

calibrate_manually(model_baseline, n_samples=32, seed=101)

# Load MICE LOO
print("Loading MICE LOO...")
mice_loo = MICEBayesianLOO(
    prior_scale=1.0, pathfinder_num_samples=100,
    pathfinder_maxiter=50, batch_size=256, verbose=True,
)
mice_loo.load('mice_loo_model.yaml')

# Build mixed imputation
print("Building mixed imputation model...")
mixed_imputation = IrtMixedImputationModel(
    irt_model=model_baseline, mice_model=mice_loo,
    data_factory=data_factory, irt_elpd_batch_size=4,
)
print(mixed_imputation.summary())

# Fit imputed GRM
print("Fitting imputed GRM...")
NUM_EPOCHS = 200

model_imputed = GRModel(
    item_keys=item_keys, num_people=SUBSAMPLE_N, dim=1,
    response_cardinality=response_cardinality,
    dtype=jnp.float32,
        parameterization="log_scale", imputation_model=mixed_imputation,
)

res_imputed = model_imputed.fit(
    data_factory, batch_size=BATCH_SIZE, dataset_size=SUBSAMPLE_N,
    num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch,
    learning_rate=1e-3, patience=10, zero_nan_grads=True,
    initial_values=snapshot_params, lr_decay_factor=0.975,
    sample_size=32, seed=43,
)
losses_imputed = res_imputed[0]
print(f"Imputed done: {losses_imputed[-1]:.2f}")
model_imputed.save_to_disk('grm_imputed')

# Need baseline losses - refit or load. We'll just use a dummy for the plot.
# Actually let's load from the baseline training - check if losses were saved
# We don't have them saved, so we'll recompute a quick eval
print("Computing baseline loss for comparison...")
losses_baseline = np.load('losses_baseline.npy')

# Generate all plots
print("Generating plots...")

fig = plot_loss_comparison(losses_baseline, losses_imputed, title='RWA Scale')
fig.savefig('loss_comparison.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

calibrate_manually(model_imputed, n_samples=32, seed=102)

fig = plot_forest_discriminations(item_keys, model_baseline, model_imputed,
                                  title='RWA — Item Discriminations')
fig.savefig('discriminations.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

ab_base = np.array(model_baseline.calibrated_expectations['abilities']).flatten()
ab_imp = np.array(model_imputed.calibrated_expectations['abilities']).flatten()

fig = plot_ability_scatter(ab_base, ab_imp, label='RWA')
fig.savefig('ability_scatter.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

fig = plot_ability_distributions(ab_base, ab_imp, label='RWA')
fig.savefig('ability_distributions.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

fig = plot_thresholds(item_keys, model_baseline, model_imputed,
                      title='RWA — Difficulty Thresholds')
fig.savefig('thresholds.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

fig = plot_individual_abilities(item_keys, model_baseline, model_imputed)
fig.savefig('individual_abilities.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

fig = plot_imputation_weights_pcolormesh(mice_loo, mixed_imputation, item_keys,
                                          title='RWA — Imputation Weights')
fig.savefig('imputation_weights.pdf', bbox_inches='tight', dpi=150)
plt.close(fig)

print("All done! Plots saved as PDFs.")
