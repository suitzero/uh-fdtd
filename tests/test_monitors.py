import unittest
import jax.numpy as jnp
from uh_fdtd.monitors.dft import init_dft_monitor, update_dft

class TestMonitors(unittest.TestCase):
    def test_dft(self):
        m = init_dft_monitor(1.0, (10,))
        m = update_dft(m, 1.0, jnp.ones(10))
        self.assertEqual(m.real_part.shape, (10,))
        self.assertEqual(m.imag_part.shape, (10,))

if __name__ == '__main__':
    unittest.main()
