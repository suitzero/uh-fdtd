import unittest
import jax.numpy as jnp
from uh_fdtd.core.grid import GridParams2D
from uh_fdtd.components.mzi import optimize_mzi

class TestMZI(unittest.TestCase):
    def test_optimize_mzi(self):
        grid_shape = (20, 20)
        params = GridParams2D(dx=0.1, dy=0.1, dt=0.05)
        eps_bg = 1.0
        eps_wg = 4.0
        source_loc = (5, 10)
        target_loc = (15, 10)
        target_phase = jnp.pi / 2  # 90 degrees phase shift
        freq = 2.0

        # Run with enough steps to get valid FDTD output
        final_density, losses = optimize_mzi(
            grid_shape=grid_shape,
            params=params,
            eps_bg=eps_bg,
            eps_wg=eps_wg,
            source_loc=source_loc,
            target_loc=target_loc,
            target_phase=target_phase,
            freq=freq,
            steps=30,
            opt_steps=5,
            learning_rate=0.1
        )

        self.assertEqual(final_density.shape, grid_shape)
        self.assertEqual(len(losses), 5)
        # Check that loss is decreasing or at least optimizing
        self.assertTrue(losses[-1] <= losses[0])

if __name__ == '__main__':
    unittest.main()
