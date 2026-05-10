import unittest
import jax.numpy as jnp
from uh_fdtd.core.grid import State1D, Material1D, GridParams1D, State2D, Material2D, GridParams2D
from uh_fdtd.core.update import run_simulation, run_simulation_2d

class TestCore(unittest.TestCase):
    def test_run_simulation_1d(self):
        state = State1D(ez=jnp.zeros(10), hy=jnp.zeros(10))
        mat = Material1D(eps=jnp.ones(10), mu=jnp.ones(10))
        params = GridParams1D(dx=1.0, dt=0.5)
        final_state, monitors = run_simulation(state, mat, params, 5)
        self.assertEqual(final_state.ez.shape, (10,))
        self.assertEqual(final_state.hy.shape, (10,))

    def test_run_simulation_2d(self):
        state = State2D(ez=jnp.zeros((10, 10)), hx=jnp.zeros((10, 10)), hy=jnp.zeros((10, 10)))
        mat = Material2D(eps=jnp.ones((10, 10)), mu=jnp.ones((10, 10)))
        params = GridParams2D(dx=1.0, dy=1.0, dt=0.5)
        final_state, monitors = run_simulation_2d(state, mat, params, 5)
        self.assertEqual(final_state.ez.shape, (10, 10))
        self.assertEqual(final_state.hx.shape, (10, 10))
        self.assertEqual(final_state.hy.shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
