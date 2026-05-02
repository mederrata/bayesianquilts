"""Bioresponse: Final attempt - proper decomposed structure with theory-based regularization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_final():
    """Bioresponse with proper decomposed structure."""
    print("\n" + "="*60)
    print("BIORESPONSE - Final: Decomposed Structure + Linear")
    print("="*60)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    N, p_orig = X.shape
    print(f"  N = {N}, p = {p_orig}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Strategy:
    # - Use Decomposed for shape functions (captures nonlinearity)
    # - Use linear term on ALL features (captures main linear signal)
    # - Exclude lattice features from linear to avoid double counting
    # - Use theory-based regularization

    n_shape_features = 4  # Top features for shape functions
    n_bins = 10  # Finer bins for better resolution

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Find most predictive features
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X_train_s, y_train)
        importance = np.abs(lr.coef_[0])
        top_features = np.argsort(importance)[::-1][:n_shape_features]
        other_features = np.array([i for i in range(p_orig) if i not in top_features])
        print(f"    Top features: {top_features.tolist()}")

        # Build dimensions for top features
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]
            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_bins = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bins = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

            dim_name = f"F{feat_idx}"
            train_indices[dim_name] = train_bins
            test_indices[dim_name] = test_bins
            dimensions.append(Dimension(dim_name, n_bins))

        total_cells = n_bins ** n_shape_features
        print(f"    Total cells: {total_cells}")

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        # Prepare features for linear term (exclude lattice features)
        X_train_other = jnp.array(X_train_s[:, other_features])
        X_test_other = jnp.array(X_test_s[:, other_features])
        y_train_j = jnp.array(y_train)

        # Use up to order 2 interactions
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(len(other_features))

        params = {"intercept": intercept_params, "beta": beta}

        # Prior variances
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            else:
                avg_count = N_train / (n_bins ** order)
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    Active components: {len(active_components)}")
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_beta:.4f}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_other * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_other * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (decomposed + linear): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_final()
