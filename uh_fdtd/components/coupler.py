import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Callable

from uh_fdtd.core.grid import State2D, Material2D, GridParams2D
from uh_fdtd.core.update import run_simulation_2d
from uh_fdtd.monitors.dft import init_dft_monitor
from uh_fdtd.sources.pulses import ContinuousWave
from uh_fdtd.sources.inject import inject_2d_ez
from uh_fdtd.optim.material import simp_interpolation
from uh_fdtd.optim.optimizer import optimize_density

def optimize_coupler(
    grid_shape: Tuple[int, int],
    params: GridParams2D,
    eps_bg: float,
    eps_wg: float,
    source_loc: Tuple[int, int],
    target_loc: Tuple[int, int],
    freq: float,
    steps: int = 50,
    opt_steps: int = 20,
    learning_rate: float = 0.1
) -> Tuple[jnp.ndarray, list[float]]:
    """
    Optimizes a 2D density distribution (e.g., a directional coupler)
    to maximize transmission from source_loc to target_loc.
    """
    cw_source = ContinuousWave(amplitude=1.0, frequency=freq)

    def source_fn(state: State2D, t: float) -> State2D:
        val = cw_source.get_value(t)
        return inject_2d_ez(state, val, source_loc[0], source_loc[1])

    def objective(density: jnp.ndarray) -> float:
        # Map density to permittivity (using SIMP to encourage binarization)
        eps = simp_interpolation(density, eps_min=eps_bg, eps_max=eps_wg)
        mu = jnp.ones(grid_shape)
        material = Material2D(eps=eps, mu=mu)

        initial_state = State2D(
            ez=jnp.zeros(grid_shape),
            hx=jnp.zeros(grid_shape),
            hy=jnp.zeros(grid_shape)
        )

        monitors = (init_dft_monitor(frequency=freq, shape=grid_shape),)

        _, final_monitors = run_simulation_2d(
            initial_state, material, params, steps=steps,
            source_fn=source_fn, monitors=monitors
        )

        # Maximize intensity at target_loc (so minimize negative intensity)
        monitor = final_monitors[0]
        intensity = monitor.real_part[target_loc]**2 + monitor.imag_part[target_loc]**2

        # Add a penalty to encourage solid structures (density close to 0 or 1)
        # Not strictly necessary, but helpful for physical designs
        # binarization_penalty = jnp.mean(density * (1.0 - density))

        return -intensity # + 0.1 * binarization_penalty

    # Start with a uniform block of material where it could be routed
    initial_density = jnp.ones(grid_shape) * 0.5

    optimizer = optax.adam(learning_rate=learning_rate)

    final_density, losses = optimize_density(
        objective, initial_density, optimizer, num_steps=opt_steps
    )

    return final_density, losses
