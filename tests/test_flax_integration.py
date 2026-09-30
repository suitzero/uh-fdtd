import unittest
import jax
import jax.numpy as jnp
from uh_fdtd.nn.mvm import optical_mvm, init_mvm_phases
from uh_fdtd.nn.flax_integration import OpticalMVM

class TestFlaxIntegration(unittest.TestCase):
    def test_forward_pass_matches_functional(self):
        """
        Verify that the Flax module's forward pass matches the functional
        optical_mvm within numerical tolerance.
        """
        key = jax.random.PRNGKey(0)
        num_mzis = 4

        # Initialize the Flax module
        module = OpticalMVM(num_mzis=num_mzis)

        # Dummy inputs (batched)
        inputs = jnp.array([
            [1.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 1.0 + 0j],
            [1.0 + 0j, 1.0 + 0j]
        ]) / jnp.sqrt(2.0)

        # Initialize parameters
        variables = module.init(key, inputs)
        phases = variables['params']['phases']

        # Forward pass using Flax
        out_flax = module.apply(variables, inputs)

        # Forward pass using functional optical_mvm directly
        out_functional = optical_mvm(phases, inputs)

        # Check numerical equivalence
        import numpy as np
        np.testing.assert_allclose(out_flax, out_functional, rtol=1e-5, atol=1e-5)

    def test_finite_gradients(self):
        """
        Compute a small scalar loss through Flax's apply/init and check for
        finite gradients using jax.value_and_grad w.r.t. the phase parameters.
        """
        key = jax.random.PRNGKey(42)
        num_mzis = 3

        module = OpticalMVM(num_mzis=num_mzis)

        # Single input vector
        inputs = jnp.array([1.0 + 0j, 0.0 + 0j])

        variables = module.init(key, inputs)

        # Define a loss function
        def loss_fn(params):
            out = module.apply({'params': params}, inputs)
            # Simple loss: squared magnitude of the first output port
            loss = jnp.sum(jnp.abs(out[0])**2)
            return loss

        loss_val, grads = jax.value_and_grad(loss_fn)(variables['params'])

        # Check that loss is finite
        self.assertTrue(jnp.isfinite(loss_val))

        # Check that gradients are finite
        grad_phases = grads['phases']
        self.assertTrue(jnp.all(jnp.isfinite(grad_phases)))

        # Check gradient shape matches phase parameters shape
        self.assertEqual(grad_phases.shape, (num_mzis, 2))

if __name__ == '__main__':
    unittest.main()
