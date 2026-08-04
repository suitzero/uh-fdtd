import jax.numpy as jnp
from typing import NamedTuple, List

from uh_fdtd.monitors.dft import DFTMonitor


class Port(NamedTuple):
    """
    Defines a physical port for S-parameter extraction.
    """
    mode_profile: jnp.ndarray


def calculate_overlap(dft_monitor: DFTMonitor, port: Port) -> jnp.complex64:
    """
    Calculates the complex mode overlap integral of the FDTD field with the port mode.

    Args:
        dft_monitor: The DFTMonitor recorded at the port.
        port: The Port definition containing the mode profile.

    Returns:
        The complex overlap amplitude.
    """
    complex_field = dft_monitor.real_part + 1j * dft_monitor.imag_part
    return jnp.vdot(port.mode_profile, complex_field)


def extract_s_parameter(
    output_monitor: DFTMonitor,
    output_port: Port,
    input_amplitude: jnp.complex64
) -> jnp.complex64:
    """
    Calculates the S-parameter (S_ji) for a single frequency.

    Args:
        output_monitor: The DFTMonitor recorded at the output port.
        output_port: The Port definition containing the expected mode profile.
        input_amplitude: The complex amplitude of the incident wave at the input port.

    Returns:
        The complex S-parameter.
    """
    overlap_out = calculate_overlap(output_monitor, output_port)
    return overlap_out / (input_amplitude + 1e-12)


def extract_s_matrix_column(
    output_monitors: List[DFTMonitor],
    output_ports: List[Port],
    input_amplitude: jnp.complex64
) -> jnp.ndarray:
    """
    Calculates a column of the S-matrix by extracting S-parameters for multiple output ports,
    given an excitation at one input port.

    Args:
        output_monitors: List of DFT monitors at the output ports.
        output_ports: List of Port definitions corresponding to the monitors.
        input_amplitude: The complex amplitude of the incident wave at the excited input port.

    Returns:
        A JAX array representing a column of the S-matrix.
    """
    s_params = [
        extract_s_parameter(monitor, port, input_amplitude)
        for monitor, port in zip(output_monitors, output_ports)
    ]
    return jnp.array(s_params)
