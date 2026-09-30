import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Any
from uh_fdtd.nn.mvm import optical_mvm, init_mvm_phases

class OpticalMVM(nn.Module):
    """
    Flax module wrapper for the optical MVM layer.
    Exposes the physical phase shift parameters (theta, phi) of the MZIs as
    trainable parameters.
    """
    num_mzis: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Initialize phases using the custom physical initializer
        phases = self.param('phases', init_mvm_phases, self.num_mzis)

        # Apply the functional optical MVM
        return optical_mvm(phases, inputs)
