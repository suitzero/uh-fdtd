import jax.numpy as jnp
from typing import Callable

def linear_interpolation(density: jnp.ndarray, eps_min: float, eps_max: float) -> jnp.ndarray:
    """
    Linearly interpolates permittivity based on density values.

    Args:
        density: An array of density values in the range [0, 1].
        eps_min: Permittivity when density is 0.
        eps_max: Permittivity when density is 1.

    Returns:
        Interpolated permittivity array.
    """
    return eps_min + density * (eps_max - eps_min)

def simp_interpolation(density: jnp.ndarray, eps_min: float, eps_max: float, penalty: float = 3.0) -> jnp.ndarray:
    """
    Solid Isotropic Material with Penalization (SIMP) interpolation.
    Used to penalize intermediate density values, encouraging binarization.

    Args:
        density: An array of density values in the range [0, 1].
        eps_min: Permittivity when density is 0.
        eps_max: Permittivity when density is 1.
        penalty: Penalization factor (typically >= 1).

    Returns:
        Interpolated permittivity array.
    """
    return eps_min + (density ** penalty) * (eps_max - eps_min)
