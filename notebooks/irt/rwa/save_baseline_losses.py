"""Quick script to refit baseline and save losses (baseline model already saved)."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import jax.numpy as jnp
from bayesianquilts.data.rwa import get_data, item_keys
from bayesianquilts.irt.grm import GRModel

response_cardinality = 9
df, num_people = get_data(polars_out=True)
SUBSAMPLE_N = num_people
BATCH_SIZE = 256
steps_per_epoch = int(np.ceil(SUBSAMPLE_N / BATCH_SIZE))

def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        data[col] = dataframe[col].to_numpy().astype(np.float64)
    data['person'] = np.arange(len(dataframe), dtype=np.float64)
    return data

batch = make_data_dict(df)

def data_factory():
    indices = np.arange(SUBSAMPLE_N)
    np.random.shuffle(indices)
    for start in range(0, SUBSAMPLE_N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, SUBSAMPLE_N)
        idx_batch = indices[start:end]
        yield {k: v[idx_batch] for k, v in batch.items()}

model = GRModel(
    item_keys=item_keys, num_people=SUBSAMPLE_N, dim=1,
    kappa_scale=0.1, response_cardinality=response_cardinality,
    dtype=jnp.float64,
)
res = model.fit(
    data_factory, batch_size=BATCH_SIZE, dataset_size=SUBSAMPLE_N,
    num_epochs=200, steps_per_epoch=steps_per_epoch,
    learning_rate=2e-4, lr_decay_factor=0.975, patience=10,
    zero_nan_grads=True, snapshot_epoch=50,
)
losses = res[0]
np.save('losses_baseline.npy', np.array(losses))
model.save_to_disk('grm_baseline')
snapshot_params = res[2] if len(res) > 2 else None
if snapshot_params is not None:
    import pickle
    with open('snapshot_params.pkl', 'wb') as f:
        pickle.dump(snapshot_params, f)
print(f"Baseline done: {losses[-1]:.2f}, saved losses and model")
