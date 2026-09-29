import unittest
import jax
import jax.numpy as jnp
from uh_fdtd.nn.mvm import (
    directional_coupler,
    waveguide_crossing,
    mzi_transfer_matrix,
    optical_mvm,
    init_mvm_phases
)

class TestMVM(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def test_directional_coupler(self):
        dc = directional_coupler()
        # Should be unitary: U^dagger @ U = I
        identity = jnp.eye(2, dtype=jnp.complex64)
        prod = jnp.conj(dc.T) @ dc
        self.assertTrue(jnp.allclose(prod, identity, atol=1e-5))

    def test_waveguide_crossing(self):
        wc = waveguide_crossing()
        inputs = jnp.array([1.0, 0.0], dtype=jnp.complex64)
        outputs = wc @ inputs
        # Should swap ports
        expected = jnp.array([0.0, 1.0], dtype=jnp.complex64)
        self.assertTrue(jnp.allclose(outputs, expected))

    def test_mzi_transfer_matrix(self):
        # MZI with 0 phase shift on both arms
        # T = PS_ext(0) * DC * PS_int(0) * DC
        # DC * DC = i * [[0, 1], [1, 0]]
        t_matrix = mzi_transfer_matrix(0.0, 0.0)

        identity = jnp.eye(2, dtype=jnp.complex64)
        prod = jnp.conj(t_matrix.T) @ t_matrix
        self.assertTrue(jnp.allclose(prod, identity, atol=1e-5))

        inputs = jnp.array([1.0, 0.0], dtype=jnp.complex64)
        outputs = t_matrix @ inputs

        # When theta=0, light crosses over completely (bar state) with pi/2 phase
        # Wait, DC * DC = 0.5 * [[1, i], [i, 1]] @ [[1, i], [i, 1]]
        # = 0.5 * [[1-1, i+i], [i+i, -1+1]] = 0.5 * [[0, 2i], [2i, 0]] = [[0, i], [i, 0]]
        expected = jnp.array([0.0, 1j], dtype=jnp.complex64)
        self.assertTrue(jnp.allclose(outputs, expected, atol=1e-5))

    def test_optical_mvm_1d(self):
        phases = jnp.array([[0.0, 0.0], [jnp.pi, 0.0]])
        inputs = jnp.array([1.0, 0.0], dtype=jnp.complex64)

        result = optical_mvm(phases, inputs)

        self.assertEqual(result.shape, (2,))
        self.assertFalse(jnp.any(jnp.isnan(result)))
        # Energy conservation (magnitude should be 1)
        power = jnp.sum(jnp.abs(result)**2)
        self.assertAlmostEqual(power, 1.0, places=5)

    def test_optical_mvm_2d_batch(self):
        phases = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi/2]])
        inputs = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)

        result = optical_mvm(phases, inputs)

        self.assertEqual(result.shape, (2, 2))
        power = jnp.sum(jnp.abs(result)**2, axis=1)
        self.assertTrue(jnp.allclose(power, jnp.ones(2), atol=1e-5))

    def test_mvm_differentiable(self):
        # We want to optimize phases to output [0, 1] given input [1, 0]
        def loss_fn(p, x):
            y = optical_mvm(p, x)
            # Maximize power at port 1 (index 1) -> minimize negative power
            return -jnp.abs(y[1])**2

        phases = init_mvm_phases(self.key, 2)
        inputs = jnp.array([1.0, 0.0], dtype=jnp.complex64)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(phases, inputs)

        self.assertEqual(grads.shape, phases.shape)
        self.assertTrue(jnp.sum(jnp.abs(grads)) > 0)

if __name__ == '__main__':
    unittest.main()
