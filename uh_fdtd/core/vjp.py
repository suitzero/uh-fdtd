from functools import partial
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, NamedTuple

from .grid import State1D, Material1D, GridParams1D, State2D, Material2D, GridParams2D
from ..monitors.dft import DFTMonitor


def reverse_step_1d(
        state: State1D,
        material: Material1D,
        params: GridParams1D) -> State1D:
    """
    Analytically reverses the 1D Yee grid update step.
    This reconstructs the state at time t from the state at time t + dt.
    """
# Reverse Ez update
    dHy = state.hy - jnp.roll(state.hy, 1)
    ez_prev = state.ez - (params.dt / (material.eps * params.dx)) * dHy

# Reverse Hy update using the reconstructed Ez
    dEz = jnp.roll(ez_prev, -1) - ez_prev
    hy_prev = state.hy - (params.dt / (material.mu * params.dx)) * dEz

    return State1D(ez=ez_prev, hy=hy_prev)


def reverse_step_2d(
        state: State2D,
        material: Material2D,
        params: GridParams2D) -> State2D:
    """
    Analytically reverses the 2D Yee grid update step.
    This reconstructs the state at time t from the state at time t + dt.
    """
# Reverse Ez update
    dHy_x = state.hy - jnp.roll(state.hy, 1, axis=0)
    dHx_y = state.hx - jnp.roll(state.hx, 1, axis=1)

    ez_prev = state.ez - (params.dt / (material.eps * params.dx)) * \
        dHy_x + (params.dt / (material.eps * params.dy)) * dHx_y

# Reverse Hx, Hy updates using reconstructed Ez
    dEz_y = jnp.roll(ez_prev, -1, axis=1) - ez_prev
    dEz_x = jnp.roll(ez_prev, -1, axis=0) - ez_prev

    hx_prev = state.hx + (params.dt / (material.mu * params.dy)) * dEz_y
    hy_prev = state.hy - (params.dt / (material.mu * params.dx)) * dEz_x

    return State2D(ez=ez_prev, hx=hx_prev, hy=hy_prev)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def run_simulation_vjp(
    initial_state: State1D,
    material: Material1D,
    params: GridParams1D,
    steps: int,
    source_fn: Callable[[State1D, float], State1D] = lambda s, t: s,
    monitors: Tuple[DFTMonitor, ...] = ()
) -> Tuple[State1D, Tuple[DFTMonitor, ...]]:

    from .update import step_1d
    from ..monitors.dft import update_dft

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
        next_field_state = step_1d(carry_sim.field_state, material, params)
        next_field_state = source_fn(next_field_state, carry_sim.time)

        next_monitors = tuple(
            update_dft(monitor, carry_sim.time, next_field_state.ez)
            for monitor in carry_sim.monitors
        )

        next_sim_state = SimState(
            field_state=next_field_state,
            monitors=next_monitors,
            time=carry_sim.time + params.dt
        )
        return next_sim_state, None

    final_sim_state, _ = jax.lax.scan(
        scan_body, initial_sim_state, None, length=steps)
    return final_sim_state.field_state, final_sim_state.monitors


def run_simulation_fwd(
    initial_state, material, params, steps, source_fn, monitors
):
    final_state, final_monitors = run_simulation_vjp(
        initial_state, material, params, steps, source_fn, monitors
    )
    return (final_state, final_monitors), (final_state,
                                           material, params, monitors)


def run_simulation_bwd(
    steps, source_fn, res, g_out
):
    final_state, material, params, monitors_init = res
    g_final_state, g_final_monitors = g_out

    from .update import step_1d

    def bwd_scan_body(carry, _):
        fwd_state, g_state, g_mat, time_prev = carry

# 1. Reverse the state (since sources are purely additive in our current implementation, we subtract
        # the source)
# Note: we assume source_fn is additive to the state (soft source).
# For a truly reverse FDTD, we first remove the source added at
# time_prev

# The source was added after step_1d, so first we subtract the source to get the state right after
        # step_1d
# To do this safely for additive sources, we'd need an inverse_source_fn, but we can't automatically
        # invert a black-box callable.
# Let's assume a generic FDTD setup where sources only affect local fields, and reverse source
        # injection.
# Actually, if the source is state-independent (e.g. state.ez.at[i].add(val)), we can just reverse it
        # by subtracting.
# But we don't have a reverse_source_fn.
# For now, let's just reverse the time step. This means sources injected are "lost" during backprop's
        # state reconstruction unless we re-simulate forward or cache.
# Given FDTD memory constraints, caching states is impossible. Re-simulating from start to end,
        # checkpointing, is an alternative (Optax/JAX does this natively with checkpointing).
# But this issue asks for *custom VJP logic to bypass memory constraints*. Analytical reversal is the
        # intended path.
# Let's just reverse the step without reversing sources for simplicity in this minimal framework, or
        # assume sources don't break reversal significantly (they do add energy, so reversing without
        # subtracting them adds errors).
# A more robust way to reverse sources is to call source_fn with negative time or negative amplitude,
        # but we can't assume its structure.
# For our specific source_fn (inject_1d_ez), it just adds.
# Actually, let's define an inverse source injection implicitly: the source adds `val` to `ez`.
# When we go backward, we can just subtract `val` from `ez`. But how to get `val`?
# We can evaluate `source_fn(zero_state, time)` to get the delta!

        zero_state = State1D(
            ez=jnp.zeros_like(
                fwd_state.ez), hy=jnp.zeros_like(
                fwd_state.hy))
        source_delta = source_fn(zero_state, time_prev)
        state_before_source = State1D(
            ez=fwd_state.ez - source_delta.ez,
            hy=fwd_state.hy - source_delta.hy
        )

        prev_fwd_state = reverse_step_1d(state_before_source, material, params)

# 2. Backpropagate through monitors update for this time step
        g_ez_from_monitors = jnp.zeros_like(fwd_state.ez)
        if monitors_init is not None:
            for i, m_init in enumerate(monitors_init):
                freq = m_init.frequency
                omega_t = 2 * jnp.pi * freq * time_prev

                g_real = g_final_monitors[i].real_part
                g_imag = g_final_monitors[i].imag_part

# gradient of (field * cos) wrt field is cos
# gradient of (field * -sin) wrt field is -sin
                g_ez_from_monitors += g_real * \
                    jnp.cos(omega_t) + g_imag * (-jnp.sin(omega_t))

# Add monitor gradients and source VJP
# Source was added to state, so VJP of source injection passes
# gradients directly to state.
        g_state_updated = State1D(
            ez=g_state.ez + g_ez_from_monitors,
            hy=g_state.hy
        )

# 3. Compute VJP for step_1d
        primals = (prev_fwd_state, material)
        def step_fn(s, m): return step_1d(s, m, params)
        _, vjp_fn = jax.vjp(step_fn, *primals)

        g_prev_state, g_step_mat = vjp_fn(g_state_updated)

# Accumulate material gradients
        g_mat_new = Material1D(
            eps=g_mat.eps + g_step_mat.eps,
            mu=g_mat.mu + g_step_mat.mu
        )

        return (prev_fwd_state, g_prev_state,
                g_mat_new, time_prev - params.dt), None

    g_mat_init = Material1D(
        eps=jnp.zeros_like(
            material.eps), mu=jnp.zeros_like(
            material.mu))

# The last time evaluated was (steps - 1) * dt
    last_time = (steps - 1) * params.dt

    init_carry = (final_state, g_final_state, g_mat_init, last_time)

    final_carry, _ = jax.lax.scan(
        bwd_scan_body, init_carry, None, length=steps)
    _, g_initial_state, g_material, _ = final_carry

    return g_initial_state, g_material, None, g_final_monitors


run_simulation_vjp.defvjp(run_simulation_fwd, run_simulation_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def run_simulation_2d_vjp(
    initial_state: State2D,
    material: Material2D,
    params: GridParams2D,
    steps: int,
    source_fn: Callable[[State2D, float], State2D] = lambda s, t: s,
    monitors: Tuple[DFTMonitor, ...] = ()
) -> Tuple[State2D, Tuple[DFTMonitor, ...]]:

    from .update import step_2d
    from ..monitors.dft import update_dft

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
        next_field_state = step_2d(carry_sim.field_state, material, params)
        next_field_state = source_fn(next_field_state, carry_sim.time)

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

    final_sim_state, _ = jax.lax.scan(
        scan_body, initial_sim_state, None, length=steps)
    return final_sim_state.field_state, final_sim_state.monitors


def run_simulation_2d_fwd(
    initial_state, material, params, steps, source_fn, monitors
):
    final_state, final_monitors = run_simulation_2d_vjp(
        initial_state, material, params, steps, source_fn, monitors
    )
    return (final_state, final_monitors), (final_state,
                                           material, params, monitors)


def run_simulation_2d_bwd(
    steps, source_fn, res, g_out
):
    final_state, material, params, monitors_init = res
    g_final_state, g_final_monitors = g_out

    from .update import step_2d

    def bwd_scan_body(carry, _):
        fwd_state, g_state, g_mat, time_prev = carry

        zero_state = State2D(
            ez=jnp.zeros_like(fwd_state.ez),
            hx=jnp.zeros_like(fwd_state.hx),
            hy=jnp.zeros_like(fwd_state.hy)
        )
        source_delta = source_fn(zero_state, time_prev)
        state_before_source = State2D(
            ez=fwd_state.ez - source_delta.ez,
            hx=fwd_state.hx - source_delta.hx,
            hy=fwd_state.hy - source_delta.hy
        )

        prev_fwd_state = reverse_step_2d(state_before_source, material, params)

        g_ez_from_monitors = jnp.zeros_like(fwd_state.ez)
        if monitors_init is not None:
            for i, m_init in enumerate(monitors_init):
                freq = m_init.frequency
                omega_t = 2 * jnp.pi * freq * time_prev

                g_real = g_final_monitors[i].real_part
                g_imag = g_final_monitors[i].imag_part

                g_ez_from_monitors += g_real * \
                    jnp.cos(omega_t) + g_imag * (-jnp.sin(omega_t))

        g_state_updated = State2D(
            ez=g_state.ez + g_ez_from_monitors,
            hx=g_state.hx,
            hy=g_state.hy
        )

        primals = (prev_fwd_state, material)
        def step_fn(s, m): return step_2d(s, m, params)
        _, vjp_fn = jax.vjp(step_fn, *primals)

        g_prev_state, g_step_mat = vjp_fn(g_state_updated)

        g_mat_new = Material2D(
            eps=g_mat.eps + g_step_mat.eps,
            mu=g_mat.mu + g_step_mat.mu
        )

        return (prev_fwd_state, g_prev_state,
                g_mat_new, time_prev - params.dt), None

    g_mat_init = Material2D(
        eps=jnp.zeros_like(
            material.eps), mu=jnp.zeros_like(
            material.mu))
    last_time = (steps - 1) * params.dt
    init_carry = (final_state, g_final_state, g_mat_init, last_time)

    final_carry, _ = jax.lax.scan(
        bwd_scan_body, init_carry, None, length=steps)
    _, g_initial_state, g_material, _ = final_carry

    return g_initial_state, g_material, None, g_final_monitors


run_simulation_2d_vjp.defvjp(run_simulation_2d_fwd, run_simulation_2d_bwd)
