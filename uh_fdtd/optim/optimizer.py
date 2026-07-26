import jax
import jax.numpy as jnp
import optax
from typing import Callable, Tuple, Any

def optimize_density(
    loss_fn: Callable[[jnp.ndarray], float],
    initial_density: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    num_steps: int
) -> Tuple[jnp.ndarray, list[float]]:
    """
    Optimizes a material density distribution using a given loss function and Optax optimizer.

    Args:
        loss_fn: A function that takes a density array and returns a scalar loss.
        initial_density: The starting density array (values should be in [0, 1]).
        optimizer: An initialized Optax optimizer (e.g., optax.adam(learning_rate=0.1)).
        num_steps: The number of optimization steps to perform.

    Returns:
        A tuple containing:
        - The optimized density array (clipped to [0, 1]).
        - A list of loss values at each step.
    """
    opt_state = optimizer.init(initial_density)

    # Define the update step using JAX
    @jax.jit
    def step(density: jnp.ndarray, opt_state: Any) -> Tuple[jnp.ndarray, Any, float]:
        loss, grads = jax.value_and_grad(loss_fn)(density)
        updates, next_opt_state = optimizer.update(grads, opt_state, density)
        next_density = optax.apply_updates(density, updates)
        # Constrain density to [0, 1] range physically meaningful for interpolation
        next_density = jnp.clip(next_density, 0.0, 1.0)
        return next_density, next_opt_state, loss

    density = initial_density
    losses = []

    for _ in range(num_steps):
        density, opt_state, loss = step(density, opt_state)
        losses.append(float(loss))

    return density, losses
