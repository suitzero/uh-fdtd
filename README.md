# uh-fdtd

A Differentiable, JAX-accelerated FDTD Engine for Optical Computing Architecture

uh-fdtd is a purely functional Finite-Difference Time-Domain (FDTD) simulator built on JAX and XLA. By leveraging auto-differentiation (Autograd) and hardware acceleration (GPU/TPU), uh-fdtd transcends traditional forward-simulation, enabling large-scale inverse design and topology optimization for nanophotonics and Optical Neural Networks (ONNs).

**Author:** Hagyoon Choi

## Core Philosophy

* **First-Principles Design:** Minimalist, zero-overhead Yee grid implementation.
* **Purely Functional:** State updates are strictly handled via `jax.lax.scan` to maintain differentiability and XLA compilation efficiency.
* **Hardware-Native:** Designed from the ground up to scale across parallel hardware architectures.

## Architecture

* **Core FDTD Engine:** 1D/2D Maxwell's equations discretized on a Yee grid.
* **Boundary Conditions:** Differentiable Convolutional Perfectly Matched Layers (CPML).
* **Adjoint Optimizer:** Memory-efficient gradient computation via custom Vector-Jacobian Products (VJP) for topology optimization.

## Planning & Roadmap

### Phase 1: JAX-Native Engine Foundation
- [x] Set up immutable state data structures for the Yee Grid.
- [x] Implement 1D/2D Maxwell curl operators using jax.numpy.
- [x] Implement pure-functional time-stepping loop (jax.lax.scan).
- [x] Integrate basic sources (Dipole, Gaussian) and DFT monitors.

### Phase 2: Differentiability & The Adjoint Method
- [x] Implement density-based material parameterization (permittivity mapping).
- [x] Build custom VJP logic to bypass memory constraints of unrolling massive time steps.
- [ ] Integrate Optax for basic gradient descent on a simple transmission loss function.

### Phase 3: Inverse Design of Optical Components
- [ ] Optimize a basic directional coupler.
- [ ] Synthesize low-loss waveguide crossings.
- [ ] Design Mach-Zehnder Interferometer (MZI) phases for active modulation.

### Phase 4: Optical Neural Network Integration
- [ ] Construct MVM (Matrix-Vector Multiplication) layers using optimized components.
- [ ] End-to-end integration with neural network frameworks (Flax/PyTorch).
