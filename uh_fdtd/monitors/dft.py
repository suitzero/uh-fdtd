import jax.numpy as jnp
from typing import NamedTuple

class DFTMonitor(NamedTuple):
    """
    State for a Discrete Fourier Transform (DFT) monitor at a specific frequency.
    """
    frequency: float
    real_part: jnp.ndarray
    imag_part: jnp.ndarray

def init_dft_monitor(frequency: float, shape: tuple) -> DFTMonitor:
    return DFTMonitor(
        frequency=frequency,
        real_part=jnp.zeros(shape),
        imag_part=jnp.zeros(shape)
    )

def update_dft(monitor: DFTMonitor, time: float, field: jnp.ndarray) -> DFTMonitor:
    """
    Updates the DFT monitor with the field value at the current time step.
    Integration uses the standard DFT kernel: field * exp(-i * 2 * pi * f * t)
    """
    omega_t = 2 * jnp.pi * monitor.frequency * time

    # exp(-i * omega * t) = cos(omega * t) - i * sin(omega * t)
    real_update = field * jnp.cos(omega_t)
    imag_update = field * (-jnp.sin(omega_t))

    return DFTMonitor(
        frequency=monitor.frequency,
        real_part=monitor.real_part + real_update,
        imag_part=monitor.imag_part + imag_update
    )
