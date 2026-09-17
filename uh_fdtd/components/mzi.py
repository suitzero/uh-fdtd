import jax
import jax.numpy as jnp
import optax
from typing import Tuple

from uh_fdtd.core.grid import State2D, Material2D, GridParams2D
from uh_fdtd.core.update import run_simulation_2d
from uh_fdtd.monitors.dft import init_dft_monitor
from uh_fdtd.sources.pulses import ContinuousWave
from uh_fdtd.sources.inject import inject_2d_ez
from uh_fdtd.optim.material import simp_interpolation
from uh_fdtd.optim.optimizer import optimize_density

def optimize_mzi(
    grid_shape: Tuple[int, int],
    params: GridParams2D,
    eps_bg: float,
    eps_wg: float,
    source_loc: Tuple[int, int],
    target_loc: Tuple[int, int],
    target_phase: float,
    freq: float,
    steps: int = 50,
    opt_steps: int = 20,
    learning_rate: float = 0.1,
    phase_weight: float = 1.0,
    transmission_weight: float = 1.0
) -> Tuple[jnp.ndarray, list[float]]:
    """
    Optimizes a 2D density distribution for a Mach-Zehnder Interferometer (MZI) phase shifter.
    Maximizes transmission from source_loc to target_loc and matches the target_phase.
    """
    cw_source = ContinuousWave(amplitude=1.0, frequency=freq)

    def source_fn(state: State2D, t: float) -> State2D:
        val = cw_source.get_value(t)
        return inject_2d_ez(state, val, source_loc[0], source_loc[1])

    def objective(density: jnp.ndarray) -> float:
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

        monitor = final_monitors[0]

        # Calculate transmission (intensity)
        target_real = monitor.real_part[target_loc]
        target_imag = monitor.imag_part[target_loc]
        intensity = target_real**2 + target_imag**2

        # Calculate phase
        phase = jnp.arctan2(target_imag, target_real)

        # Phase error (circular distance)
        phase_diff = jnp.angle(jnp.exp(1j * (phase - target_phase)))
        phase_error = phase_diff**2

        # Objective is to minimize negative transmission (maximize it) and minimize phase error
        return -transmission_weight * intensity + phase_weight * phase_error

    # Start with a uniform block of material
    initial_density = jnp.ones(grid_shape) * 0.5

    optimizer = optax.adam(learning_rate=learning_rate)

    final_density, losses = optimize_density(
        objective, initial_density, optimizer, num_steps=opt_steps
    )

    return final_density, losses
