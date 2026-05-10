import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Any, NamedTuple

from .grid import State1D, Material1D, GridParams1D, State2D, Material2D, GridParams2D
from ..monitors.dft import DFTMonitor, update_dft

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
    steps: int,
    source_fn: Callable[[State1D, float], State1D] = lambda s, t: s,
    monitors: Tuple[DFTMonitor, ...] = ()
) -> Tuple[State1D, Tuple[DFTMonitor, ...]]:
    """
    Runs the 1D FDTD simulation for a given number of steps using jax.lax.scan.
    This maintains differentiability and allows efficient XLA compilation without loop unrolling.
    """
    class SimState(NamedTuple):
        field_state: State1D
        monitors: Tuple[DFTMonitor, ...]
        time: float

    initial_sim_state = SimState(
        field_state=initial_state,
        monitors=monitors,
        time=0.0
    )

    def scan_body(carry_sim: SimState, _):
        # 1. Update H and E fields
        next_field_state = step_1d(carry_sim.field_state, material, params)

        # 2. Inject sources
        next_field_state = source_fn(next_field_state, carry_sim.time)

        # 3. Update Monitors
        next_monitors = tuple(
            update_dft(monitor, carry_sim.time, next_field_state.ez)
            for monitor in carry_sim.monitors
        )

        next_sim_state = SimState(
            field_state=next_field_state,
            monitors=next_monitors,
            time=carry_sim.time + params.dt
        )
        # Returns the carried state, and None for the accumulated history
        return next_sim_state, None

    final_sim_state, _ = jax.lax.scan(scan_body, initial_sim_state, None, length=steps)
    return final_sim_state.field_state, final_sim_state.monitors

def update_h_2d(state: State2D, material: Material2D, params: GridParams2D) -> State2D:
    """
    Updates Magnetic fields (Hx, Hy) for 2D TM mode.
    Periodic boundary conditions.
    """
    dEz_y = jnp.roll(state.ez, -1, axis=1) - state.ez
    dEz_x = jnp.roll(state.ez, -1, axis=0) - state.ez

    hx_new = state.hx - (params.dt / (material.mu * params.dy)) * dEz_y
    hy_new = state.hy + (params.dt / (material.mu * params.dx)) * dEz_x

    return State2D(ez=state.ez, hx=hx_new, hy=hy_new)

def update_e_2d(state: State2D, material: Material2D, params: GridParams2D) -> State2D:
    """
    Updates Electric field (Ez) for 2D TM mode.
    Periodic boundary conditions.
    """
    dHy_x = state.hy - jnp.roll(state.hy, 1, axis=0)
    dHx_y = state.hx - jnp.roll(state.hx, 1, axis=1)

    ez_new = state.ez + (params.dt / (material.eps * params.dx)) * dHy_x - (params.dt / (material.eps * params.dy)) * dHx_y

    return State2D(ez=ez_new, hx=state.hx, hy=state.hy)

def step_2d(state: State2D, material: Material2D, params: GridParams2D) -> State2D:
    """
    Performs a single full FDTD time step for 2D (Yee algorithm).
    """
    state = update_h_2d(state, material, params)
    state = update_e_2d(state, material, params)
    return state

def run_simulation_2d(
    initial_state: State2D,
    material: Material2D,
    params: GridParams2D,
    steps: int,
    source_fn: Callable[[State2D, float], State2D] = lambda s, t: s,
    monitors: Tuple[DFTMonitor, ...] = ()
) -> Tuple[State2D, Tuple[DFTMonitor, ...]]:

    class SimState2D(NamedTuple):
        field_state: State2D
        monitors: Tuple[DFTMonitor, ...]
        time: float

    initial_sim_state = SimState2D(
        field_state=initial_state,
        monitors=monitors,
        time=0.0
    )

    def scan_body(carry_sim: SimState2D, _):
        # 1. Update H and E fields
        next_field_state = step_2d(carry_sim.field_state, material, params)

        # 2. Inject sources
        next_field_state = source_fn(next_field_state, carry_sim.time)

        # 3. Update Monitors
        next_monitors = tuple(
            update_dft(monitor, carry_sim.time, next_field_state.ez)
            for monitor in carry_sim.monitors
        )

        next_sim_state = SimState2D(
            field_state=next_field_state,
            monitors=next_monitors,
            time=carry_sim.time + params.dt
        )
        return next_sim_state, None

    final_sim_state, _ = jax.lax.scan(scan_body, initial_sim_state, None, length=steps)
    return final_sim_state.field_state, final_sim_state.monitors
