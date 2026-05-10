import jax.numpy as jnp
from ..core.grid import State1D, State2D

def inject_1d_ez(state: State1D, source_value: float, index: int) -> State1D:
    """Injects a soft source into the Ez field at a specific index."""
    new_ez = state.ez.at[index].add(source_value)
    return State1D(ez=new_ez, hy=state.hy)

def inject_2d_ez(state: State2D, source_value: float, x_idx: int, y_idx: int) -> State2D:
    """Injects a soft source into the Ez field at a specific 2D index."""
    new_ez = state.ez.at[x_idx, y_idx].add(source_value)
    return State2D(ez=new_ez, hx=state.hx, hy=state.hy)
