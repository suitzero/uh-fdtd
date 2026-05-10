import jax.numpy as jnp
from typing import NamedTuple

class GaussianPulse(NamedTuple):
    amplitude: float
    frequency: float
    width: float
    delay: float

    def get_value(self, time: float) -> float:
        # standard gaussian pulse: A * exp(-((t - delay) / width)^2) * sin(2*pi*f*t)
        envelope = jnp.exp(-((time - self.delay) / self.width)**2)
        carrier = jnp.sin(2 * jnp.pi * self.frequency * time)
        return self.amplitude * envelope * carrier

class ContinuousWave(NamedTuple):
    amplitude: float
    frequency: float

    def get_value(self, time: float) -> float:
        return self.amplitude * jnp.sin(2 * jnp.pi * self.frequency * time)
