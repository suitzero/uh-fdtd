import jax
import jax.numpy as jnp
from typing import Tuple, List

# --- Component Transfer Matrices (S-parameters) ---

def phase_shifter(theta: float) -> jnp.ndarray:
    """
    Transfer matrix (S-matrix) of a single waveguide phase shifter.
    Applies a phase shift e^(i*theta).
    """
    return jnp.exp(1j * theta)

def directional_coupler() -> jnp.ndarray:
    """
    Transfer matrix of an ideal 50/50 directional coupler.
    (1/sqrt(2)) * [[1, i], [i, 1]]
    """
    return (1.0 / jnp.sqrt(2.0)) * jnp.array([
        [1.0, 1j],
        [1j, 1.0]
    ], dtype=jnp.complex64)

def waveguide_crossing() -> jnp.ndarray:
    """
    Transfer matrix of an ideal waveguide crossing (no crosstalk).
    Swaps the two ports.
    """
    return jnp.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=jnp.complex64)

def mzi_transfer_matrix(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Constructs the transfer matrix of a Mach-Zehnder Interferometer (MZI)
    using directional couplers and phase shifters (extracted S-parameters).

    Structure:
    - Input 50/50 Coupler
    - Phase Shifter on one arm (theta)
    - Output 50/50 Coupler
    - External phase shifter on one output (phi)

    Args:
        theta: Internal phase shift.
        phi: External phase shift.

    Returns:
        2x2 complex unitary transfer matrix.
    """
    dc = directional_coupler()

    # Internal phase shifts: e^(i*theta) on top arm, 1 on bottom arm
    ps_internal = jnp.array([
        [jnp.exp(1j * theta), 0.0],
        [0.0, 1.0]
    ], dtype=jnp.complex64)

    # External phase shifts: e^(i*phi) on top arm, 1 on bottom arm
    ps_external = jnp.array([
        [jnp.exp(1j * phi), 0.0],
        [0.0, 1.0]
    ], dtype=jnp.complex64)

    # Cascade: M = PS_ext * DC * PS_int * DC
    # Using matmul (@)
    return ps_external @ dc @ ps_internal @ dc

def cascaded_mzi_mesh(phases: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Simulates a simple cascaded line of MZIs.
    This acts as our Optical MVM layer constructed from physical components.

    Args:
        phases: A (N, 2) array of (theta, phi) pairs for each MZI.
        inputs: A (2,) complex array representing the input optical field.

    Returns:
        A (2,) complex array representing the output optical field.
    """
    def apply_mzi(carry, phase_pair):
        theta, phi = phase_pair[0], phase_pair[1]
        t_matrix = mzi_transfer_matrix(theta, phi)
        out = t_matrix @ carry
        return out, None

    # We use scan to differentiable cascade the components
    final_out, _ = jax.lax.scan(apply_mzi, inputs, phases)
    return final_out

def optical_mvm(phases: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Optical Matrix-Vector Multiplication layer using MZI components.

    Args:
        phases: An array of phase shifts representing the weights.
                For this basic implementation, shape is (N, 2) where N
                is the number of cascaded 2x2 MZIs.
        inputs: Input vector, complex amplitudes.
                If batched, shape is (B, 2). If unbatched, shape is (2,).

    Returns:
        Output vector, same shape as inputs.
    """
    if inputs.ndim == 1:
        return cascaded_mzi_mesh(phases, inputs)
    elif inputs.ndim == 2:
        # Vectorize over the batch dimension
        return jax.vmap(cascaded_mzi_mesh, in_axes=(None, 0))(phases, inputs)
    else:
        raise ValueError("Inputs must be 1D (vector) or 2D (batch of vectors)")

def init_mvm_phases(key: jax.Array, num_mzis: int) -> jnp.ndarray:
    """
    Initializes the physical phase settings (theta, phi) for the MVM layer.
    """
    return jax.random.uniform(key, shape=(num_mzis, 2), minval=0.0, maxval=2*jnp.pi)
