import unittest
import jax.numpy as jnp
from uh_fdtd.core.grid import GridParams2D
from uh_fdtd.components.crossing import optimize_crossing

class TestCrossing(unittest.TestCase):
    def test_optimize_crossing(self):
        # Use a very small grid and few steps for a quick unit test
        grid_shape = (10, 10)
        params = GridParams2D(dx=0.1, dy=0.1, dt=0.05)

        eps_bg = 1.0
        eps_wg = 4.0

        source_loc = (2, 5)
        target_loc = (8, 5)
        crosstalk_locs = [(5, 2), (5, 8)]
        freq = 2.0

        final_density, losses = optimize_crossing(
            grid_shape=grid_shape,
            params=params,
            eps_bg=eps_bg,
            eps_wg=eps_wg,
            source_loc=source_loc,
            target_loc=target_loc,
            crosstalk_locs=crosstalk_locs,
            freq=freq,
            steps=10,        # Very few steps just to check the pipeline runs
            opt_steps=3,     # Few optimization steps
            learning_rate=0.1,
            crosstalk_weight=1.0
        )

        self.assertEqual(final_density.shape, grid_shape)
        self.assertEqual(len(losses), 3)
        self.assertTrue(jnp.all(final_density >= 0.0) and jnp.all(final_density <= 1.0))
        # Check it runs without error.

if __name__ == '__main__':
    unittest.main()