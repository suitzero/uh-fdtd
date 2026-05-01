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
    Spatial and temporal discretization parameters for 1D.
    """
    dx: float
    dt: float

class State2D(NamedTuple):
    """
    Immutable state for a 2D Yee Grid (TM mode: Ez, Hx, Hy).
    Represents the fields at a given time step.
    """
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray

class Material2D(NamedTuple):
    """
    Material properties for a 2D Yee Grid.
    Supports spatially varying permittivity (eps) and permeability (mu).
    """
    eps: jnp.ndarray
    mu: jnp.ndarray

class GridParams2D(NamedTuple):
    """
    Spatial and temporal discretization parameters for 2D.
    """
    dx: float
    dy: float
    dt: float
