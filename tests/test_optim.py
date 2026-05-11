import unittest
import jax.numpy as jnp
from uh_fdtd.optim.material import linear_interpolation, simp_interpolation

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

if __name__ == '__main__':
    unittest.main()
