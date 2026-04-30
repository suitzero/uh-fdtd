import jax
import jax.numpy as jnp
from typing import Tuple

from .grid import State1D, Material1D, GridParams1D

def update_h(state: State1D, material: Material1D, params: GridParams1D) -> State1D:
    """
    Updates the Magnetic field (Hy) for one half time step.
    Assumes periodic boundary conditions using jnp.roll for simplicity in this minimal core.
    """
    # spatial derivative of Ez
    dEz = jnp.roll(state.ez, -1) - state.ez

    # Update Hy
    hy_new = state.hy + (params.dt / (material.mu * params.dx)) * dEz

    return State1D(ez=state.ez, hy=hy_new)

def update_e(state: State1D, material: Material1D, params: GridParams1D) -> State1D:
    """
    Updates the Electric field (Ez) for one half time step.
    Assumes periodic boundary conditions using jnp.roll for simplicity in this minimal core.
    """
    # spatial derivative of Hy
    dHy = state.hy - jnp.roll(state.hy, 1)

    # Update Ez
    ez_new = state.ez + (params.dt / (material.eps * params.dx)) * dHy

    return State1D(ez=ez_new, hy=state.hy)

def step_1d(state: State1D, material: Material1D, params: GridParams1D) -> State1D:
    """
    Performs a single full FDTD time step (Yee algorithm).
    """
    state = update_h(state, material, params)
    state = update_e(state, material, params)
    return state

def run_simulation(
    initial_state: State1D,
    material: Material1D,
    params: GridParams1D,
    steps: int
) -> State1D:
    """
    Runs the 1D FDTD simulation for a given number of steps using jax.lax.scan.
    This maintains differentiability and allows efficient XLA compilation without loop unrolling.
    """
    def scan_body(carry_state: State1D, _):
        next_state = step_1d(carry_state, material, params)
        # Returns the carried state, and None for the accumulated history (to save memory)
        return next_state, None

    final_state, _ = jax.lax.scan(scan_body, initial_state, None, length=steps)
    return final_state
