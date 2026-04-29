#!/usr/bin/env python
"""Run marginal MCMC for all 3 model variants: baseline, pairwise, mixed.

For each variant, loads the fitted ADVI baseline GRM, optionally attaches
an imputation model, then runs BlackJAX NUTS on item parameters with
abilities Rao-Blackwellized out on a Gauss-Hermite quadrature grid.

Results are saved per-variant:
  - grm_mcmc_baseline/    (no imputation)
  - grm_mcmc_pairwise/    (pairwise stacking imputation)
  - grm_mcmc_mixed/       (mixed: pairwise + IRT baseline)

Usage:
    python run_marginal_mcmc.py --dataset rwa
    python run_marginal_mcmc.py --dataset tma --num-chains 2 --num-warmup 200
    python run_marginal_mcmc.py --dataset npi --variants baseline pairwise
"""

import argparse
import gc
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np
import jax
import jax.numpy as jnp


DATASET_CONFIGS = {
    'scs': {'module': 'bayesianquilts.data.scs'},
    'gcbs': {'module': 'bayesianquilts.data.gcbs'},
    'grit': {'module': 'bayesianquilts.data.grit'},
    'rwa': {'module': 'bayesianquilts.data.rwa'},
    'npi': {'module': 'bayesianquilts.data.npi'},
    'tma': {'module': 'bayesianquilts.data.tma'},
    'wpi': {'module': 'bayesianquilts.data.wpi'},
    'eqsq': {'module': 'bayesianquilts.data.eqsq'},
    'promis_sleep': {'module': 'bayesianquilts.data.promis_sleep'},
    'promis_substance_use': {'module': 'bayesianquilts.data.promis_substance_use'},
    'promis_neuropathic_pain': {'module': 'bayesianquilts.data.promis_neuropathic_pain', 'factorized': True},
    'promis_copd': {'module': 'bayesianquilts.data.promis_copd', 'factorized': True},
    # Per-domain unidim GRMs (replace factorized PROMIS fits)
    'promis_np__pain_interference':  {'module': 'bayesianquilts.data.promis_neuropathic_pain', 'domain': 'pain_interference'},
    'promis_np__pain_behavior':      {'module': 'bayesianquilts.data.promis_neuropathic_pain', 'domain': 'pain_behavior'},
    'promis_np__global_health':      {'module': 'bayesianquilts.data.promis_neuropathic_pain', 'domain': 'global_health'},
    'promis_copd__depression':         {'module': 'bayesianquilts.data.promis_copd', 'domain': 'depression'},
    'promis_copd__anxiety':            {'module': 'bayesianquilts.data.promis_copd', 'domain': 'anxiety'},
    'promis_copd__anger':              {'module': 'bayesianquilts.data.promis_copd', 'domain': 'anger'},
    'promis_copd__fatigue_experience': {'module': 'bayesianquilts.data.promis_copd', 'domain': 'fatigue_experience'},
    'promis_copd__fatigue_impact':     {'module': 'bayesianquilts.data.promis_copd', 'domain': 'fatigue_impact'},
    'promis_copd__pain_interference':  {'module': 'bayesianquilts.data.promis_copd', 'domain': 'pain_interference'},
    'promis_copd__pain_behavior':      {'module': 'bayesianquilts.data.promis_copd', 'domain': 'pain_behavior'},
    'promis_copd__physical_function':  {'module': 'bayesianquilts.data.promis_copd', 'domain': 'physical_function'},
    'promis_copd__social_satisfaction':{'module': 'bayesianquilts.data.promis_copd', 'domain': 'social_satisfaction'},
    # PROMIS Substance Use sub-banks (correlation-derived, Ward; max 50 items per bank)
    'promis_su__bank1': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank1'},
    'promis_su__bank2': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank2'},
    'promis_su__bank3': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank3'},
    'promis_su__bank4': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank4'},
    'promis_su__bank5': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank5'},
    'promis_su__bank6': {'module': 'bayesianquilts.data.promis_substance_use', 'bank': 'bank6'},
    # PROMIS 1 Wave 1 calibration (14 health domains; planned-missing-by-form)
    'promis_w1__alcohol_use':         {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'alcohol_use'},
    'promis_w1__anger':               {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'anger'},
    'promis_w1__anxiety':             {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'anxiety'},
    'promis_w1__depression':          {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'depression'},
    'promis_w1__fatigue_experience':  {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'fatigue_experience'},
    'promis_w1__fatigue_impact':      {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'fatigue_impact'},
    'promis_w1__pain_behavior':       {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'pain_behavior'},
    'promis_w1__pain_interference':   {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'pain_interference'},
    'promis_w1__pain_quality':        {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'pain_quality'},
    'promis_w1__physical_function_a': {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'physical_function_a'},
    'promis_w1__physical_function_b': {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'physical_function_b'},
    'promis_w1__physical_function_c': {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'physical_function_c'},
    'promis_w1__social_personal':     {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'social_personal'},
    'promis_w1__social_satisfaction': {'module': 'bayesianquilts.data.promis_wave1', 'domain': 'social_satisfaction'},
}


def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float32)
        data[col] = arr
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def run_single_variant(model, data, variant_name, output_dir,
                       num_chains, num_warmup, num_samples, step_size, seed,
                       sampler='nuts', dense_mass=False):
    """Run MCMC for one variant and save results."""
    print(f"\n  --- Variant: {variant_name} (sampler={sampler}) ---")
    sys.stdout.flush()

    if sampler == 'mala':
        mcmc_samples = model.fit_marginal_mala(
            data,
            theta_grid=None,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            step_size=step_size,
            seed=seed,
            verbose=True,
        )
    else:
        mcmc_samples = model.fit_marginal_mcmc(
            data,
            theta_grid=None,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_accept_prob=0.85,
            step_size=step_size,
            seed=seed,
            verbose=True,
            dense_mass=dense_mass,
        )

    # EAP
    print(f"\n  Computing EAP abilities...")
    eap_result = model.compute_eap_abilities(data)
    print(f"    EAP mean: {np.mean(np.array(eap_result['eap'])):.4f}")
    print(f"    EAP std:  {np.std(np.array(eap_result['eap'])):.4f}")
    print(f"    Mean PSD: {np.mean(np.array(eap_result['psd'])):.4f}")

    # Standardize
    print(f"\n  Standardizing...")
    stats = model.standardize_marginal(data)
    eap_result = model.compute_eap_abilities(data)
    print(f"    Post-std EAP std: {np.std(np.array(eap_result['eap'])):.4f}")

    # Fit surrogate
    model.fit_surrogate_to_mcmc()

    # Inject EAP abilities
    eap_arr = np.array(eap_result['eap'])
    model.surrogate_sample['abilities'] = jnp.array(
        eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
    )[np.newaxis, ...]

    # Save model
    model_dir = output_dir / f'grm_mcmc_{variant_name}'
    print(f"  Saving to {model_dir}")
    model.save_to_disk(str(model_dir))

    # Save NPZ
    npz_dir = output_dir / 'mcmc_samples'
    os.makedirs(npz_dir, exist_ok=True)
    save_dict = {}
    for var_name, samples in mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    save_dict['eap'] = np.array(eap_result['eap'])
    save_dict['psd'] = np.array(eap_result['psd'])
    save_dict['standardize_mu'] = stats['mu']
    save_dict['standardize_sigma'] = stats['sigma']
    npz_path = npz_dir / f'mcmc_{variant_name}.npz'
    np.savez(str(npz_path), **save_dict)
    print(f"  NPZ: {npz_path}")

    # Summary
    print(f"\n  --- {variant_name} Summary ---")
    for var_name, samples in mcmc_samples.items():
        flat = np.array(samples).reshape(-1, *samples.shape[2:])
        print(f"    {var_name}: mean={np.mean(flat):.4f}, std={np.std(flat):.4f}")
        if samples.shape[0] > 1:
            s = np.array(samples)
            chain_means = np.mean(s, axis=1)
            between_var = np.var(chain_means, axis=0, ddof=1)
            within_var = np.mean(np.var(s, axis=1, ddof=1), axis=0)
            n = samples.shape[1]
            # Flag params whose chains barely moved — typical of step-size
            # floor hits where R-hat numerically explodes despite chains
            # being effectively stationary at (close to) the init. Use a
            # relative-to-between-var threshold so absolute-scale tiny
            # params don't over-report as "frozen".
            eps = np.maximum(between_var * 1e-8, np.abs(chain_means).mean() * 1e-12)
            frozen = within_var <= eps
            frac_frozen = float(np.mean(frozen))
            r_hat = np.sqrt(
                ((n - 1) / n * within_var + between_var) /
                np.maximum(within_var, np.maximum(eps, 1e-30))
            )
            # Report R-hat on non-frozen params (informative) and frozen
            # fraction separately (diagnostic).
            if frac_frozen < 1.0:
                rh_ok = r_hat[~frozen]
                print(f"      R-hat (non-frozen): mean={np.mean(rh_ok):.4f}, "
                      f"max={np.max(rh_ok):.4f}, frac<=1.1={(rh_ok<=1.1).mean():.2f}")
            if frac_frozen > 0:
                print(f"      frac_frozen (within_var≈0): {frac_frozen:.2f}")
    sys.stdout.flush()


def run_dataset(dataset_name, model_dir, num_chains, num_warmup, num_samples,
                step_size, seed, variants, sampler='nuts', dense_mass=False,
                mcmc_disc_prior_scale=None, mcmc_ddiff_prior_scale=None,
                mcmc_d0_prior_scale=None):
    import importlib
    import inspect
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel

    def _apply_prior_overrides(model):
        if mcmc_disc_prior_scale is not None:
            model.mcmc_disc_prior_scale = mcmc_disc_prior_scale
        if mcmc_ddiff_prior_scale is not None:
            model.mcmc_ddiff_prior_scale = mcmc_ddiff_prior_scale
        if mcmc_d0_prior_scale is not None:
            model.mcmc_d0_prior_scale = mcmc_d0_prior_scale
        return model

    config = DATASET_CONFIGS[dataset_name]
    is_factorized = config.get('factorized', False)
    domain = config.get('domain')
    bank = config.get('bank')
    mod = importlib.import_module(config['module'])
    response_cardinality = mod.response_cardinality

    if is_factorized:
        from bayesianquilts.irt.factorizedgrm import FactorizedGRModel as ModelClass
    else:
        ModelClass = GRModel

    model_path = Path(model_dir)
    output_dir = model_path.parent

    # Load data first (per-domain loaders set ``mod.item_keys`` only inside get_data)
    if is_factorized and hasattr(mod, 'get_multidomain_data'):
        df, num_people, scale_indices = mod.get_multidomain_data(
            polars_out=True, min_items=10)
    else:
        sig = inspect.signature(mod.get_data).parameters
        get_data_kwargs = {'polars_out': True}
        if 'reorient' in sig:
            get_data_kwargs['reorient'] = True
        if domain is not None and 'domain' in sig:
            get_data_kwargs['domain'] = domain
        if bank is not None and 'bank' in sig:
            get_data_kwargs['bank'] = bank
        df, num_people = mod.get_data(**get_data_kwargs)
    item_keys = list(mod.item_keys)
    base_data = make_data_dict(df, num_people)

    sub_label = domain or bank
    print(f"\n{'='*60}")
    print(f"Marginal MCMC: {dataset_name.upper()}"
          + (f"  [{sub_label}]" if sub_label else ""))
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  People: {num_people}")
    print(f"  Chains: {num_chains}, Warmup: {num_warmup}, Samples: {num_samples}")
    print(f"  Step size: {step_size}")
    print(f"  Variants: {variants}")
    print(f"{'='*60}")

    # Load pairwise stacking model (needed for pairwise and mixed)
    pairwise_model = None
    stacking_path = output_dir / 'pairwise_stacking_model.yaml'
    if stacking_path.exists() and ('pairwise' in variants or 'mixed' in variants):
        from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
        print(f"  Loading pairwise stacking model...")
        pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))

    # ---- Baseline: no imputation ----
    if 'baseline' in variants:
        model = _apply_prior_overrides(ModelClass.load_from_disk(str(model_path)))
        data = dict(base_data)  # no imputation PMFs
        run_single_variant(model, data, 'baseline', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed,
                           sampler=sampler, dense_mass=dense_mass)
        del model
        gc.collect()

    # ---- Pairwise: pairwise stacking only ----
    if 'pairwise' in variants and pairwise_model is not None:
        model = _apply_prior_overrides(ModelClass.load_from_disk(str(model_path)))
        model.imputation_model = pairwise_model
        data = dict(base_data)
        pmfs, weights = model._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
            # For pairwise-only, don't pass weights (full imputation, no IS blend)
        print(f"  Pairwise imputation PMFs attached")
        run_single_variant(model, data, 'pairwise', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed + 1,
                           sampler=sampler, dense_mass=dense_mass)
        del model
        gc.collect()

    # ---- Mixed: pairwise + IRT baseline blend ----
    if 'mixed' in variants and pairwise_model is not None:
        from bayesianquilts.imputation.mixed import IrtMixedImputationModel
        model = _apply_prior_overrides(ModelClass.load_from_disk(str(model_path)))

        # Build mixed imputation model from pairwise + baseline
        # Need calibrated expectations for the IRT component
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed + 100)
        samples = surrogate.sample(32, seed=key)
        model.surrogate_sample = samples
        model.calibrated_expectations = {
            k: jnp.mean(v, axis=0) for k, v in samples.items()
        }

        def _data_factory():
            yield base_data

        mixed_model = IrtMixedImputationModel(
            irt_model=model,
            mice_model=pairwise_model,
            data_factory=_data_factory,
        )
        model.imputation_model = mixed_model

        data = dict(base_data)
        pmfs, _ = model._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
        print(f"  Mixed imputation PMFs attached")
        run_single_variant(model, data, 'mixed', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed + 2,
                           sampler=sampler, dense_mass=dense_mass)
        del model, mixed_model
        gc.collect()

    del pairwise_model
    gc.collect()
    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run marginal MCMC for all model variants')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--model-dir', default=None,
                        help='Path to grm_baseline/ directory (default: auto)')
    parser.add_argument('--num-chains', type=int, default=2)
    parser.add_argument('--num-warmup', type=int, default=200)
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--step-size', type=float, default=5e-4,
                        help='Initial NUTS step size')
    parser.add_argument('--variants', nargs='+',
                        default=['baseline', 'pairwise', 'mixed'],
                        choices=['baseline', 'pairwise', 'mixed'],
                        help='Which model variants to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampler', default='nuts', choices=['nuts', 'mala'],
                        help='MCMC sampler: nuts (default) or mala (for stiff posteriors)')
    parser.add_argument('--dense-mass', action='store_true',
                        help='Use dense mass matrix (default: diagonal). '
                             'NUTS only; ignored for MALA.')
    parser.add_argument('--mcmc-disc-prior-scale', type=float, default=None,
                        help='Override MCMC discrimination prior scale '
                             '(HalfNormal). Default 0.5; tighter (e.g. 0.2) '
                             'helps multimodal real-data posteriors.')
    parser.add_argument('--mcmc-ddiff-prior-scale', type=float, default=None,
                        help='Override MCMC ddifficulties prior scale (HalfN).')
    parser.add_argument('--mcmc-d0-prior-scale', type=float, default=None,
                        help='Override MCMC difficulties0 prior scale '
                             '(Normal at d0_loc). Default 0.3.')
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{args.dataset}/grm_baseline')

    if not os.path.exists(os.path.join(model_dir, 'params.h5')):
        print(f"ERROR: No params.h5 found in {model_dir}")
        sys.exit(1)

    run_dataset(
        args.dataset,
        model_dir,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        step_size=args.step_size,
        seed=args.seed,
        variants=args.variants,
        sampler=args.sampler,
        dense_mass=args.dense_mass,
        mcmc_disc_prior_scale=args.mcmc_disc_prior_scale,
        mcmc_ddiff_prior_scale=args.mcmc_ddiff_prior_scale,
        mcmc_d0_prior_scale=args.mcmc_d0_prior_scale,
    )


if __name__ == '__main__':
    main()
