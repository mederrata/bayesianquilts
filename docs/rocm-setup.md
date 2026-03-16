# ROCm GPU Setup for Strix Halo (AMD Radeon 8060S)

This guide covers setting up bayesianquilts with JAX on AMD Radeon 8060S (gfx1151, RDNA 3.5) via ROCm on Ubuntu 24.04.

## Prerequisites

- Ubuntu 24.04 (Noble)
- AMD Strix Halo APU with Radeon 8060S (gfx1151)
- amdgpu driver installed (v6.4+ via `amdgpu-install`)
- Python 3.12 (ROCm JAX wheels not yet available for 3.13)
- uv package manager

## 1. Install amdgpu driver (if not already installed)

```bash
sudo amdgpu-install --usecase=dkms
```

Verify the GPU is detected:

```bash
lspci | grep -i display
# Should show: Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 1586
```

## 2. Install ROCm runtime

The JAX ROCm 7 plugin requires `rocprofiler-sdk` from the ROCm 7.2 repository. You can keep the 6.4 amdgpu driver but need the ROCm 7.2 apt repo for runtime libraries.

```bash
# Add ROCm 7.2 repo (keep the amdgpu driver repo at 6.4)
sudo sed -i 's|rocm/apt/6.4|rocm/apt/7.2|g' /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocprofiler-sdk
```

Verify ROCm sees the GPU:

```bash
rocm-smi
rocminfo | grep gfx
# Should show: gfx1151
```

## 3. Create the Python environment

```bash
cd /path/to/bayesianquilts/python

# Must use Python 3.12 (not 3.13)
uv venv --python 3.12 .venv-rocm

# Install JAX with ROCm 7 plugin
VIRTUAL_ENV=.venv-rocm uv pip install \
    jax jaxlib jax-rocm7-pjrt jax-rocm7-plugin \
    -f https://github.com/ROCm/jax/releases

# Install TFP (must use nightly for JAX 0.9.x compatibility)
VIRTUAL_ENV=.venv-rocm uv pip install tfp-nightly

# Install remaining dependencies and bayesianquilts
VIRTUAL_ENV=.venv-rocm uv pip install \
    flax optax orbax-checkpoint arviz pandas tqdm \
    -e .
```

## 4. Verify

```bash
.venv-rocm/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

print('Backend:', jax.default_backend())  # Should print: gpu
print('Devices:', jax.devices())          # Should show RocmDevice

x = jnp.ones((1000, 1000), dtype=jnp.float64)
y = (x @ x).block_until_ready()
print('Matmul OK:', float(y[0,0]))
"
```

Expected output:

```
Backend: gpu
Devices: [RocmDevice(id=0)]
Matmul OK: 1000.0
```

## 5. Run bayesianquilts on GPU

```bash
.venv-rocm/bin/python -c "
import jax; jax.config.update('jax_enable_x64', True)
from bayesianquilts.metrics.ais import LogisticRegressionLikelihood, AdaptiveImportanceSampler
print('Backend:', jax.default_backend())
print('Import OK')
"
```

## Troubleshooting

### `librocprofiler-sdk-attach.so.1: cannot open shared object file`

The ROCm 7 runtime is not installed. Install `rocprofiler-sdk` from the ROCm 7.2 repo (step 2).

### `rocm_plugin_extension not found` / falls back to CPU

The ROCm plugin can't find the HIP runtime. Check:

```bash
ls /opt/rocm/lib/libamdhip64.so*
ldconfig -p | grep rocprofiler-sdk-attach
```

### Segfault on import

Usually a ROCm version mismatch. The gfx1151 architecture requires ROCm 6.3+ runtime. The ROCm 6.0 JAX plugin (`jax-rocm60-*`) will segfault on this GPU — use `jax-rocm7-*` only.

### `jax_rocm7_plugin version X is not compatible with jaxlib version Y`

JAX and the ROCm plugin versions must match exactly. Install both at the same version:

```bash
VIRTUAL_ENV=.venv-rocm uv pip install jax==0.9.1 jaxlib==0.9.1 jax-rocm7-pjrt==0.9.1 jax-rocm7-plugin==0.9.1
```

### TFP `AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`

TFP stable (0.25.0) is incompatible with JAX 0.9.x. Use `tfp-nightly`:

```bash
VIRTUAL_ENV=.venv-rocm uv pip install tfp-nightly
```

### GPU slower than CPU for small problems

Expected for small S and N. GPU overhead (kernel launch, memory transfer) dominates. The GPU advantage appears for larger posterior sample sizes (S > 4000) or larger datasets.

## Version matrix (tested)

| Component | Version |
|-----------|---------|
| Ubuntu | 24.04 |
| amdgpu driver | 6.4 (amdgpu-dkms) |
| ROCm runtime | 7.2 (rocprofiler-sdk) |
| Python | 3.12.3 |
| JAX | 0.9.1 |
| jax-rocm7-pjrt | 0.9.1 |
| jax-rocm7-plugin | 0.9.1 |
| TFP | 0.26.0-dev (nightly) |
| GPU | Radeon 8060S (gfx1151) |
