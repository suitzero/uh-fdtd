import unittest
import jax.numpy as jnp
from uh_fdtd.monitors.dft import init_dft_monitor, update_dft, DFTMonitor
from uh_fdtd.monitors.s_parameters import Port, calculate_overlap, extract_s_parameter, extract_s_matrix_column

class TestMonitors(unittest.TestCase):
    def test_dft(self):
        m = init_dft_monitor(1.0, (10,))
        m = update_dft(m, 1.0, jnp.ones(10))
        self.assertEqual(m.real_part.shape, (10,))
        self.assertEqual(m.imag_part.shape, (10,))

    def test_s_parameters(self):
        # Create dummy DFT monitors
        monitor1 = DFTMonitor(frequency=1.0, real_part=jnp.array([1.0, 0.0]), imag_part=jnp.array([0.0, 1.0]))
        monitor2 = DFTMonitor(frequency=1.0, real_part=jnp.array([0.0, 1.0]), imag_part=jnp.array([1.0, 0.0]))

        # Create dummy ports
        port1 = Port(mode_profile=jnp.array([1.0, 0.0]))
        port2 = Port(mode_profile=jnp.array([0.0, 1.0]))

        # Test calculate_overlap
        # monitor1 complex field: [1+0j, 0+1j]
        # port1 profile: [1, 0]
        # overlap = 1 * (1+0j) + 0 * (0+1j) = 1+0j
        overlap1 = calculate_overlap(monitor1, port1)
        self.assertTrue(jnp.isclose(overlap1, 1.0 + 0.0j))

        # Test extract_s_parameter
        # S-parameter = overlap / input_amplitude
        s_param = extract_s_parameter(monitor1, port1, input_amplitude=2.0 + 0.0j)
        self.assertTrue(jnp.isclose(s_param, 0.5 + 0.0j))

        # Test extract_s_matrix_column
        # overlap of monitor2 with port2 = 0 * (0+1j) + 1 * (1+0j) = 1+0j
        s_col = extract_s_matrix_column([monitor1, monitor2], [port1, port2], input_amplitude=1.0 + 0.0j)
        self.assertEqual(s_col.shape, (2,))
        self.assertTrue(jnp.isclose(s_col[0], 1.0 + 0.0j))
        self.assertTrue(jnp.isclose(s_col[1], 1.0 + 0.0j))

if __name__ == '__main__':
    unittest.main()
