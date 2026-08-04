import unittest
import jax.numpy as jnp
import optax
from uh_fdtd.optim.material import linear_interpolation, simp_interpolation
from uh_fdtd.optim.objective import transmission_loss
from uh_fdtd.optim.optimizer import optimize_density
from uh_fdtd.monitors.dft import DFTMonitor, init_dft_monitor
from uh_fdtd.core.grid import State1D, Material1D, GridParams1D
from uh_fdtd.core.update import run_simulation
from uh_fdtd.sources.inject import inject_1d_ez
from uh_fdtd.sources.pulses import ContinuousWave

class TestMaterialOptim(unittest.TestCase):
    def test_linear_interpolation(self):
        density = jnp.array([0.0, 0.5, 1.0])
        eps_min = 1.0
        eps_max = 3.0

        expected = jnp.array([1.0, 2.0, 3.0])
        result = linear_interpolation(density, eps_min, eps_max)

        self.assertTrue(jnp.allclose(result, expected))

    def test_simp_interpolation(self):
        density = jnp.array([0.0, 0.5, 1.0])
        eps_min = 1.0
        eps_max = 3.0
        penalty = 3.0

        expected = jnp.array([1.0, 1.0 + (0.5**3) * 2.0, 3.0])
        result = simp_interpolation(density, eps_min, eps_max, penalty)

        self.assertTrue(jnp.allclose(result, expected))

    def test_transmission_loss(self):
        # Create a dummy monitor with known intensity
        monitor = DFTMonitor(
            frequency=1.0,
            real_part=jnp.array([1.0, 2.0]),
            imag_part=jnp.array([0.0, -1.0])
        )
        # Intensity: [1^2 + 0^2, 2^2 + (-1)^2] = [1, 5]
        # Sum = 6. Negative sum = -6
        loss = transmission_loss(monitor)
        self.assertAlmostEqual(loss, -6.0)

    def test_optimize_density_basic(self):
        # A simple dummy loss function that wants density to be close to 0.7
        def dummy_loss(density):
            return jnp.sum((density - 0.7)**2)

        initial_density = jnp.array([0.1, 0.9, 0.5])
        optimizer = optax.adam(learning_rate=0.1)

        final_density, losses = optimize_density(dummy_loss, initial_density, optimizer, num_steps=50)

        self.assertTrue(len(losses) == 50)
        self.assertTrue(losses[-1] < losses[0])
        self.assertTrue(jnp.allclose(final_density, 0.7, atol=0.05))

    def test_e2e_optimization_1d(self):
        # Setting up a simple 1D optimization problem
        grid_size = 20
        params = GridParams1D(dx=0.1, dt=0.05)

        # Source definition
        source_cw = ContinuousWave(amplitude=1.0, frequency=2.0)
        source_idx = 5
        def source_fn(state, time):
            val = source_cw.get_value(time)
            return inject_1d_ez(state, val, source_idx)

        # Monitor definition
        monitor_idx = 15
        freq = 2.0

        # Function to compute loss given density
        def objective(density):
            # Map density to permittivity
            eps = linear_interpolation(density, eps_min=1.0, eps_max=4.0)
            mu = jnp.ones(grid_size)
            material = Material1D(eps=eps, mu=mu)

            initial_state = State1D(ez=jnp.zeros(grid_size), hy=jnp.zeros(grid_size))

            # We want to measure power at monitor_idx
            # Since update_dft computes over the whole array, we'll slice it in loss
            monitors = (init_dft_monitor(frequency=freq, shape=(grid_size,)),)

            _, final_monitors = run_simulation(
                initial_state, material, params, steps=30,
                source_fn=source_fn, monitors=monitors
            )

            # Loss is negative intensity at monitor index
            intensity = final_monitors[0].real_part[monitor_idx]**2 + final_monitors[0].imag_part[monitor_idx]**2
            return -intensity

        initial_density = jnp.ones(grid_size) * 0.5
        optimizer = optax.adam(learning_rate=0.05)

        final_density, losses = optimize_density(objective, initial_density, optimizer, num_steps=20)

        # We expect loss (negative intensity) to decrease (i.e. intensity increases)
        self.assertTrue(losses[-1] < losses[0])

if __name__ == '__main__':
    unittest.main()
