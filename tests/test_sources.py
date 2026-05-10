import unittest
import jax.numpy as jnp
from uh_fdtd.sources.pulses import GaussianPulse, ContinuousWave
from uh_fdtd.sources.inject import inject_1d_ez, inject_2d_ez
from uh_fdtd.core.grid import State1D, State2D

class TestSources(unittest.TestCase):
    def test_pulses(self):
        p = GaussianPulse(1.0, 1.0, 1.0, 1.0)
        self.assertTrue(isinstance(p.get_value(1.0), float) or isinstance(p.get_value(1.0), jnp.ndarray))

        cw = ContinuousWave(1.0, 1.0)
        self.assertTrue(isinstance(cw.get_value(1.0), float) or isinstance(cw.get_value(1.0), jnp.ndarray))

    def test_inject(self):
        state1d = State1D(ez=jnp.zeros(10), hy=jnp.zeros(10))
        state1d = inject_1d_ez(state1d, 1.0, 5)
        self.assertEqual(state1d.ez[5], 1.0)

        state2d = State2D(ez=jnp.zeros((10, 10)), hx=jnp.zeros((10, 10)), hy=jnp.zeros((10, 10)))
        state2d = inject_2d_ez(state2d, 1.0, 5, 5)
        self.assertEqual(state2d.ez[5, 5], 1.0)

if __name__ == '__main__':
    unittest.main()
