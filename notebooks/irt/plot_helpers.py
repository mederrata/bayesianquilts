"""Shared plotting helpers for IRT GRM notebooks.

All figures follow Tufte principles:
- High data-ink ratio
- No chartjunk
- Scatter/dot plots over bar charts
- Minimal gridlines
- Clear, direct labeling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Consistent style across all notebooks
STYLE = {
    'baseline_color': '#4477AA',   # blue
    'imputed_color': '#EE6677',    # red/coral
    'mice_color': '#228833',       # green
    'irt_color': '#CCBB44',        # yellow
    'true_color': '#222222',       # near-black
    'marker_baseline': 'o',
    'marker_imputed': 's',
    'marker_size': 4,
    'capsize': 2,
    'elinewidth': 1,
    'alpha': 0.7,
    'figsize_forest': (6, None),   # width fixed, height varies
    'figsize_scatter': (5, 5),
    'figsize_loss': (7, 3),
    'dpi': 150,
}


def _set_tufte_style(ax):
    """Apply Tufte-style formatting to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


def plot_loss_comparison(losses_baseline, losses_imputed, title=None, ax=None):
    """Plot training loss curves for baseline and imputed models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=STYLE['figsize_loss'])
    else:
        fig = ax.figure

    ax.plot(losses_baseline, color=STYLE['baseline_color'], alpha=0.8,
            linewidth=1.2, label='Baseline (ignorable)')
    ax.plot(losses_imputed, color=STYLE['imputed_color'], alpha=0.8,
            linewidth=1.2, label='Imputed (Rao-Blackwell)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (neg ELBO)')
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _set_tufte_style(ax)
    plt.tight_layout()
    return fig


def plot_forest_discriminations(item_keys, model_baseline, model_imputed,
                                 title='Item Discriminations', ax=None):
    """Forest plot of discrimination parameters with posterior uncertainty."""
    disc_base = np.array(model_baseline.surrogate_sample['discriminations']).reshape(
        -1, len(item_keys))
    disc_imp = np.array(model_imputed.surrogate_sample['discriminations']).reshape(
        -1, len(item_keys))

    n_items = len(item_keys)
    height = max(4, n_items * 0.3)
    if ax is None:
        fig, ax = plt.subplots(figsize=(STYLE['figsize_forest'][0], height))
    else:
        fig = ax.figure

    y_pos = np.arange(n_items)
    offset = 0.15

    ax.errorbar(disc_base.mean(0), y_pos - offset, xerr=disc_base.std(0),
                fmt=STYLE['marker_baseline'], capsize=STYLE['capsize'],
                markersize=STYLE['marker_size'], elinewidth=STYLE['elinewidth'],
                color=STYLE['baseline_color'], alpha=STYLE['alpha'],
                label='Baseline')
    ax.errorbar(disc_imp.mean(0), y_pos + offset, xerr=disc_imp.std(0),
                fmt=STYLE['marker_imputed'], capsize=STYLE['capsize'],
                markersize=STYLE['marker_size'], elinewidth=STYLE['elinewidth'],
                color=STYLE['imputed_color'], alpha=STYLE['alpha'],
                label='Imputed')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(item_keys, fontsize=max(5, 9 - n_items // 20))
    ax.set_xlabel('Discrimination')
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    _set_tufte_style(ax)
    plt.tight_layout()
    return fig


def plot_ability_scatter(abilities_baseline, abilities_imputed, label='latent trait',
                         ax=None):
    """Scatter plot of baseline vs imputed ability estimates."""
    ab_base = np.array(abilities_baseline).flatten()
    ab_imp = np.array(abilities_imputed).flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=STYLE['figsize_scatter'])
    else:
        fig = ax.figure

    ax.scatter(ab_base, ab_imp, alpha=0.15, s=6, edgecolors='none',
               color=STYLE['baseline_color'])
    lims = [min(ab_base.min(), ab_imp.min()), max(ab_base.max(), ab_imp.max())]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=0.5, label='y = x')
    ax.set_xlabel('Baseline ability')
    ax.set_ylabel('Imputed ability')
    ax.set_title(f'Ability estimates ({label})', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_aspect('equal')
    _set_tufte_style(ax)
    plt.tight_layout()
    return fig


def plot_ability_distributions(abilities_baseline, abilities_imputed,
                                label='latent trait', ax=None):
    """Step histograms of ability distributions."""
    ab_base = np.array(abilities_baseline).flatten()
    ab_imp = np.array(abilities_imputed).flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.hist(ab_base, bins=30, histtype='step', linewidth=1.5,
            label='Baseline', color=STYLE['baseline_color'])
    ax.hist(ab_imp, bins=30, histtype='step', linewidth=1.5,
            label='Imputed', color=STYLE['imputed_color'])
    ax.set_xlabel(f'Ability ({label})')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)
    _set_tufte_style(ax)
    plt.tight_layout()
    return fig


def plot_thresholds(item_keys, model_baseline, model_imputed, title=None):
    """Panel plot of difficulty thresholds per level."""
    def _compute_thresholds(model):
        diff0 = np.array(model.surrogate_sample['difficulties0'])
        ddiff = np.array(model.surrogate_sample['ddifficulties'])
        d0 = np.concatenate([diff0, ddiff], axis=-1)
        thresholds = np.cumsum(d0, axis=-1)
        n_samples = thresholds.shape[0]
        n_thresholds = thresholds.size // (n_samples * len(item_keys))
        return thresholds.reshape(n_samples, len(item_keys), n_thresholds)

    thresh_base = _compute_thresholds(model_baseline)
    thresh_imp = _compute_thresholds(model_imputed)

    n_items = len(item_keys)
    K_minus_1 = thresh_base.shape[-1]
    n_cols = min(4, K_minus_1)
    n_rows = int(np.ceil(K_minus_1 / n_cols))
    row_height = max(4, n_items * 0.2)

    fig, axes = plt.subplots(n_rows, n_cols,
        figsize=(4.5 * n_cols, row_height * n_rows), squeeze=False)

    for level in range(K_minus_1):
        ax = axes[level // n_cols, level % n_cols]
        y_pos = np.arange(n_items)
        offset = 0.15

        ax.errorbar(thresh_base[:, :, level].mean(0), y_pos - offset,
                     xerr=thresh_base[:, :, level].std(0),
                     fmt=STYLE['marker_baseline'], capsize=2, markersize=3,
                     elinewidth=0.8, color=STYLE['baseline_color'],
                     alpha=STYLE['alpha'], label='Baseline')
        ax.errorbar(thresh_imp[:, :, level].mean(0), y_pos + offset,
                     xerr=thresh_imp[:, :, level].std(0),
                     fmt=STYLE['marker_imputed'], capsize=2, markersize=3,
                     elinewidth=0.8, color=STYLE['imputed_color'],
                     alpha=STYLE['alpha'], label='Imputed')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(item_keys, fontsize=max(4, 8 - n_items // 20))
        ax.set_title(f'Threshold {level + 1}', fontsize=10)
        ax.invert_yaxis()
        _set_tufte_style(ax)
        if level == 0:
            ax.legend(frameon=False, fontsize=7)

    for idx in range(K_minus_1, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_individual_abilities(item_keys, model_baseline, model_imputed,
                               n_show=100, seed=42):
    """Forest plot of abilities for a random subset of individuals."""
    ab_base = np.array(model_baseline.surrogate_sample['abilities']).reshape(
        model_baseline.surrogate_sample['abilities'].shape[0], -1)
    ab_imp = np.array(model_imputed.surrogate_sample['abilities']).reshape(
        model_imputed.surrogate_sample['abilities'].shape[0], -1)

    N = ab_base.shape[1]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=min(n_show, N), replace=False))

    mean_b, std_b = ab_base[:, idx].mean(0), ab_base[:, idx].std(0)
    mean_i, std_i = ab_imp[:, idx].mean(0), ab_imp[:, idx].std(0)

    order = np.argsort(mean_b)
    mean_b, std_b = mean_b[order], std_b[order]
    mean_i, std_i = mean_i[order], std_i[order]

    fig, ax = plt.subplots(figsize=(6, min(25, n_show * 0.22)))
    y_pos = np.arange(len(idx))
    offset = 0.15

    ax.errorbar(mean_b, y_pos - offset, xerr=std_b,
                fmt=STYLE['marker_baseline'], capsize=1, markersize=2,
                elinewidth=0.7, color=STYLE['baseline_color'],
                alpha=STYLE['alpha'], label='Baseline')
    ax.errorbar(mean_i, y_pos + offset, xerr=std_i,
                fmt=STYLE['marker_imputed'], capsize=1, markersize=2,
                elinewidth=0.7, color=STYLE['imputed_color'],
                alpha=STYLE['alpha'], label='Imputed')

    ax.set_yticks(y_pos[::5])
    ax.set_yticklabels([f'{i}' for i in idx[order][::5]], fontsize=5)
    ax.set_xlabel('Ability')
    ax.set_title(f'Individual abilities ({len(idx)} persons)', fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.invert_yaxis()
    _set_tufte_style(ax)
    plt.tight_layout()
    return fig


def plot_imputation_weights_pcolormesh(mice_model, mixed_model, item_keys,
                                        title='Imputation Model Weights'):
    """Pcolormesh visualization of imputation model weights.

    Y-axis: variable being predicted
    X-axis: predictor item (MICE models) + rightmost column for IRT model
    Color: model weight contribution

    Args:
        mice_model: Fitted MICEBayesianLOO model
        mixed_model: IrtMixedImputationModel with computed weights
        item_keys: List of item names
    """
    n_items = len(item_keys)

    # Build weight matrix: (n_items, n_items + 1)
    # Columns 0..n_items-1 = MICE predictor weights
    # Column n_items = IRT weight
    weight_matrix = np.zeros((n_items, n_items + 1))

    # For each target variable, compute MICE predictor weights
    for i, target_key in enumerate(item_keys):
        target_idx = i

        # Get the MICE weight for this item from the mixed model
        w_mice = mixed_model._weights.get(target_key, 0.5)
        w_irt = 1.0 - w_mice

        # Get ELPD values for each predictor to compute relative predictor weights
        predictor_elpds = {}
        for j in range(n_items):
            if i == j:
                continue
            key = (target_idx, j)
            if key in mice_model.univariate_results:
                result = mice_model.univariate_results[key]
                if result.converged:
                    predictor_elpds[j] = result.elpd_loo_per_obs

        # Add zero-predictor baseline
        if target_idx in mice_model.zero_predictor_results:
            zp = mice_model.zero_predictor_results[target_idx]
            if zp.converged:
                zero_elpd = zp.elpd_loo_per_obs
            else:
                zero_elpd = -np.inf
        else:
            zero_elpd = -np.inf

        # Compute softmax weights over predictors (within MICE)
        if predictor_elpds:
            elpd_vals = np.array(list(predictor_elpds.values()))
            pred_indices = list(predictor_elpds.keys())

            # Include zero-predictor as baseline
            all_elpds = np.concatenate([[zero_elpd], elpd_vals])

            # Softmax for numerical stability
            max_e = np.max(all_elpds)
            exp_e = np.exp(all_elpds - max_e)
            softmax_w = exp_e / exp_e.sum()

            # softmax_w[0] is the zero-predictor weight (not shown separately)
            # Distribute the total MICE weight among predictors
            for k, j in enumerate(pred_indices):
                # Weight = (mice total weight) * (predictor's share)
                weight_matrix[i, j] = w_mice * softmax_w[k + 1]

        # IRT weight in the last column
        weight_matrix[i, n_items] = w_irt

    # Create the plot
    fig, ax = plt.subplots(figsize=(max(6, (n_items + 2) * 0.35),
                                     max(4, n_items * 0.3)))

    # Use a diverging colormap centered on meaningful values
    # Sequential is better here since weights are 0-1
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=weight_matrix.max())

    # pcolormesh
    x_labels = list(item_keys) + ['IRT']
    im = ax.pcolormesh(weight_matrix, cmap=cmap, norm=norm, edgecolors='white',
                        linewidth=0.5)

    # Axis labels
    ax.set_xticks(np.arange(n_items + 1) + 0.5)
    ax.set_xticklabels(x_labels, rotation=90,
                        fontsize=max(5, 8 - n_items // 20))
    ax.set_yticks(np.arange(n_items) + 0.5)
    ax.set_yticklabels(item_keys, fontsize=max(5, 8 - n_items // 20))

    ax.set_xlabel('Predictor (MICE models) → IRT')
    ax.set_ylabel('Target variable')
    ax.set_title(title, fontsize=11)

    # Add a vertical line before the IRT column
    ax.axvline(x=n_items, color='black', linewidth=1.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Weight', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Mask diagonal (self-prediction)
    for i in range(n_items):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                    facecolor='lightgray', edgecolor='white',
                                    linewidth=0.5))

    _set_tufte_style(ax)
    plt.tight_layout()
    return fig
