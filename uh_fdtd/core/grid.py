from typing import NamedTuple
import jax.numpy as jnp

class State1D(NamedTuple):
    """
    Immutable state for a 1D Yee Grid.
    Represents the Electric (Ez) and Magnetic (Hy) fields at a given time step.
    """
    ez: jnp.ndarray
    hy: jnp.ndarray

class Material1D(NamedTuple):
    """
    Material properties for a 1D Yee Grid.
    Supports spatially varying permittivity (eps) and permeability (mu).
    """
    eps: jnp.ndarray
    mu: jnp.ndarray

class GridParams1D(NamedTuple):
    """
    Spatial and temporal discretization parameters.
    """
    dx: float
    dt: float
