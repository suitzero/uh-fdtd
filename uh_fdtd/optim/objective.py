import jax.numpy as jnp
from ..monitors.dft import DFTMonitor

def transmission_loss(monitor: DFTMonitor) -> float:
    """
    Computes a simple transmission loss based on the intensity of a DFT monitor.
    Minimizing this loss maximizes the transmission (intensity) at the monitor location.

    Args:
        monitor: A DFTMonitor object containing the frequency-domain fields.

    Returns:
        The negative sum of the intensity (real**2 + imag**2) across the monitor.
    """
    intensity = monitor.real_part**2 + monitor.imag_part**2
    return -jnp.sum(intensity)
