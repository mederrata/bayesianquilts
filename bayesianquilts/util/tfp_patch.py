import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.internal import auto_composite_tensor
from tensorflow_probability.substrates.jax import tf2jax as tf
from jax import tree_util
import functools

def fixed_pytree_flatten(obj):
    """Flatten method for JAX pytrees (Fixed for TFP bug)."""
    # pylint: disable=protected-access
    components = obj._type_spec._to_components(obj)
    if components:
        keys, values = zip(*components.items())
    else:
        keys, values = (), ()
        
    # FIX: Handle missing _structure_with_callables
    structure_with_callables = getattr(obj._type_spec, '_structure_with_callables', None)
    
    metadata = dict(
        non_tensor_params=obj._type_spec._non_tensor_params,
        structure_with_callables=structure_with_callables)
    return values, (keys, metadata)

def _unflatten_model_fixed(components, structure_with_callables):
    # This mimics the logic in TFP's _unflatten_model but handles None structure_with_callables
    # If structure_with_callables is None, we assume straightforward mapping (no callables hidden)
    if structure_with_callables is None:
        return components['model']

    model_components = []
    i = 0
    for c in tf.nest.flatten(structure_with_callables):
        if c is None:
            model_components.append(components['model'][i])
            i += 1
        else:
            model_components.append(c)
    return tf.nest.pack_sequence_as(structure_with_callables, model_components)


def fixed_pytree_unflatten(cls, aux_data, children):
    keys, metadata = aux_data
    model_dists = dict(list(zip(keys, children)))
    
    # FIX: Use our fixed unflatten model function or handle the logic here
    structure_with_callables = metadata.get('structure_with_callables')
    
    if structure_with_callables is None:
         # If no structure stored, fallback or assuming simple model components
         # In the case of missing attribute (the bug), components['model'] should be what we want
         # if it was packed by _to_components
         if 'model' in model_dists:
             model = model_dists['model']
         else:
             # This path might need adjustment based on how keys formed
             # But if keys came from components.items(), and to_components returns dict(model=...)
             # then model_dists should have 'model'.
             model = model_dists
    else:
         model = _unflatten_model_fixed(model_dists, structure_with_callables)

    return cls(model, **metadata['non_tensor_params'])

# Apply the patch
tree_util.register_pytree_node(
    tfd.JointDistributionNamed,
    fixed_pytree_flatten,
    functools.partial(fixed_pytree_unflatten, tfd.JointDistributionNamed))

tree_util.register_pytree_node(
    tfd.JointDistributionNamedAutoBatched,
    fixed_pytree_flatten,
    functools.partial(fixed_pytree_unflatten, tfd.JointDistributionNamedAutoBatched))

print("Applied TFP JointDistributionNamed PyTree patch.")
