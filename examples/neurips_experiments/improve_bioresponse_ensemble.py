"""Bioresponse: Ensemble of multiple lattice structures."""
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


def run_bioresponse_ensemble():
    """Bioresponse with ensemble of multiple lattice structures."""
    print("\n" + "="*60)
    print("BIORESPONSE - Ensemble of Lattice Structures")
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

    # Find categorical features
    cat_features = []
    cat_levels = {}
    for i in range(X.shape[1]):
        unique_vals = np.unique(X[:, i])
        if len(unique_vals) <= 10:
            cat_features.append(i)
            cat_levels[i] = unique_vals

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

        # Select top continuous features
        cont_features = [i for i in range(p_orig) if i not in cat_features]
        cont_importance = [(i, importance[i]) for i in cont_features]
        cont_importance.sort(key=lambda x: -x[1])
        top_cont = [idx for idx, imp in cont_importance[:6]]

        # Select top categorical features
        cat_importance = [(i, importance[i], len(cat_levels[i])) for i in cat_features]
        cat_importance.sort(key=lambda x: -x[1])
        top_cat = [(idx, n_levels) for idx, imp, n_levels in cat_importance[:4] if n_levels >= 2]

        print(f"    Top continuous: {top_cont[:3]}")
        print(f"    Top categorical: {[(f'D{idx+1}', n) for idx, n in top_cat[:3]]}")

        # Build 3 different lattice structures:
        # 1. Top 3 continuous features (6 bins each)
        # 2. Top 3 categorical features
        # 3. Mixed: 1 continuous + 2 categorical

        n_bins = 6
        all_logits = []

        # === Lattice 1: Top continuous features ===
        dimensions1 = []
        train_indices1 = {}
        test_indices1 = {}
        for feat_idx in top_cont[:3]:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]
            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_bins = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bins = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)
            dim_name = f"C{feat_idx}"
            train_indices1[dim_name] = train_bins
            test_indices1[dim_name] = test_bins
            dimensions1.append(Dimension(dim_name, n_bins))

        # === Lattice 2: Top categorical features ===
        dimensions2 = []
        train_indices2 = {}
        test_indices2 = {}
        for feat_idx, n_levels in top_cat[:3]:
            unique_vals = cat_levels[feat_idx]
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            train_labels = np.array([val_to_idx.get(v, 0) for v in X_train[:, feat_idx]])
            test_labels = np.array([val_to_idx.get(v, 0) for v in X_test[:, feat_idx]])
            dim_name = f"D{feat_idx+1}"
            train_indices2[dim_name] = train_labels
            test_indices2[dim_name] = test_labels
            dimensions2.append(Dimension(dim_name, n_levels))

        # === Lattice 3: Mixed ===
        dimensions3 = []
        train_indices3 = {}
        test_indices3 = {}
        # 1 continuous
        feat_idx = top_cont[0]
        train_vals = X_train_s[:, feat_idx]
        test_vals = X_test_s[:, feat_idx]
        edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
        train_bins = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
        test_bins = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)
        train_indices3[f"C{feat_idx}"] = train_bins
        test_indices3[f"C{feat_idx}"] = test_bins
        dimensions3.append(Dimension(f"C{feat_idx}", n_bins))
        # 2 categorical
        for feat_idx, n_levels in top_cat[:2]:
            unique_vals = cat_levels[feat_idx]
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            train_labels = np.array([val_to_idx.get(v, 0) for v in X_train[:, feat_idx]])
            test_labels = np.array([val_to_idx.get(v, 0) for v in X_test[:, feat_idx]])
            dim_name = f"D{feat_idx+1}"
            train_indices3[dim_name] = train_labels
            test_indices3[dim_name] = test_labels
            dimensions3.append(Dimension(dim_name, n_levels))

        # Train all lattices together with shared beta
        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Theory priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        # Initialize all lattices
        lattices = []
        for dims, train_idx, test_idx in [
            (dimensions1, train_indices1, test_indices1),
            (dimensions2, train_indices2, test_indices2),
            (dimensions3, train_indices3, test_indices3),
        ]:
            interactions = Interactions(dimensions=dims)
            decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")
            dim_names = [d.name for d in decomp._interactions._dimensions]
            train_int = jnp.stack([jnp.array(train_idx[name]) for name in dim_names], axis=-1)
            test_int = jnp.stack([jnp.array(test_idx[name]) for name in dim_names], axis=-1)
            active = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]

            # Prior vars
            prior_vars = {}
            total_cells = 1
            for d in dims:
                total_cells *= d.cardinality
            for name in active:
                order = decomp.component_order(name)
                if order == 0:
                    prior_vars[name] = tau_global ** 2
                else:
                    avg_count = N_train / (total_cells ** (order / len(dims)))
                    tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                    prior_vars[name] = tau ** 2

            lattices.append({
                'decomp': decomp,
                'train_int': train_int,
                'test_int': test_int,
                'active': active,
                'prior_vars': prior_vars,
            })

        # Initialize params
        params = {
            'beta': jnp.zeros(p_orig),
            'weights': jnp.ones(3) / 3,  # Ensemble weights
        }
        for i, lat in enumerate(lattices):
            params[f'lat{i}'] = {name: jnp.zeros(lat['decomp']._tensor_part_shapes[name]) for name in lat['active']}

        def loss_fn(params):
            # Compute weighted ensemble of lattice intercepts
            ensemble_intercept = jnp.zeros(N_train)
            weights = jax.nn.softmax(params['weights'])

            for i, lat in enumerate(lattices):
                int_vals = lat['decomp'].lookup_flat(lat['train_int'], params[f'lat{i}'])
                ensemble_intercept += weights[i] * int_vals[:, 0]

            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + ensemble_intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Regularization
            reg = 0.0
            for i, lat in enumerate(lattices):
                for name, param in params[f'lat{i}'].items():
                    var = lat['prior_vars'].get(name, 1.0)
                    reg += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            return bce + (reg + l2_beta) / N_train

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
                weights = jax.nn.softmax(params['weights'])
                print(f"    Step {i}: loss = {loss:.4f}, weights = {np.array(weights)}")

        # Evaluate
        weights = jax.nn.softmax(params['weights'])
        ensemble_intercept = jnp.zeros(len(y_test))
        for i, lat in enumerate(lattices):
            int_vals = lat['decomp'].lookup_flat(lat['test_int'], params[f'lat{i}'])
            ensemble_intercept += weights[i] * int_vals[:, 0]

        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + ensemble_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (ensemble): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_ensemble()
