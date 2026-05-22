"""bayesianquilts: Adaptive Importance Sampling and quilted-model scaling.

Top-level imports are kept thin so the bare package loads without the
optional JAX / TFP / Stan toolchain. Symbols like ``BayesianModel`` and
``MICELogistic`` are reachable via the same canonical paths they always
were (``bayesianquilts.model.BayesianModel`` etc.) and via lazy
``__getattr__`` attribute access on this module.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.9.0"

_LAZY = {
    "BayesianModel": ("bayesianquilts.model", "BayesianModel"),
    "MICELogistic": ("bayesianquilts.imputation", "MICELogistic"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'bayesianquilts' has no attribute {name!r}")
    mod_name, attr = target
    module = importlib.import_module(mod_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


if TYPE_CHECKING:  # static type checkers see the real names
    from .model import BayesianModel
    from .imputation import MICELogistic
