import tensorflow_probability.substrates.jax.distributions.joint_distribution_named as jdn
import tensorflow_probability.substrates.jax.distributions.joint_distribution_auto_batched as jdab

def patch_spec_class(cls):
    """Adds _structure_with_callables to TFP Spec classes if missing."""
    if not hasattr(cls, '_structure_with_callables'):
        # We set it as a property that returns None if the instance doesn't have it
        # However, simply setting a class attribute None works if the instance attribute
        # is missing, as python lookups fall back to class.
        # But wait, if __slots__ is used, we might have issues.
        # Let's check if __slots__ prevents this.
        # Typically TFP specs use __slots__.
        
        # If __slots__ is present and doesn't include the name, we can't set it on instance
        # unless we modify the class. But we can't add slots to a compiled class.
        # But we CAN add a property to the class that returns None.
        
        # Checking if property works.
        setattr(cls, '_structure_with_callables', None)
        print(f"Patched {cls.__name__} with _structure_with_callables=None")

try:
    patch_spec_class(jdn._JointDistributionNamedSpec)
    patch_spec_class(jdab._JointDistributionNamedAutoBatchedSpec)
    print("TFP Specs patched successfully.")
except Exception as e:
    print(f"Failed to patch TFP specs: {e}")
