# Photonics Simulation Platform

A Scalable, JAX-accelerated FDTD Engine & EDA Pipeline for Optical Computing Architectures.

This platform bridges the gap between device-level electromagnetic simulations and circuit-level behavioral modeling, enabling large-scale inverse design and topology optimization for nanophotonics and Optical Neural Networks (ONNs).

**Author:** Hagyoon Choi

## Core Philosophy

* **Unified Simulation Pipeline:** Seamlessly extract S-parameters from the native FDTD solver and inject them into a Mach-Zehnder Interferometer (MZI) mesh simulator.
* **First-Principles Design:** Minimalist, zero-overhead Yee grid implementation.
* **Purely Functional:** State updates are strictly handled via `jax.lax.scan` to maintain differentiability and XLA compilation efficiency.
* **Hardware-Native & Scalable:** Designed from the ground up to scale across parallel hardware architectures (GPU/TPU) and Kubernetes clusters for distributed job scheduling.

## Architecture Layers

**1. Core Simulation Engines**
* **uh-fdtd:** 1D/2D Maxwell's equations discretized on a Yee grid, with differentiable CPML boundaries. Resolves device-level (e.g., couplers, phase shifters) Maxwell physics.
* **MZI Circuit Simulator:** Simulates ideal combinations of transfer matrices on a circuit level.

**2. Platform Layer**
* A Kubernetes-based orchestration layer designed for distributed execution.
* Go-based Job Scheduler & API Server.
* NATS queueing and GPU worker autoscaling.
* GitOps deployment via Terraform & ArgoCD, with observability via Prometheus & OpenTelemetry.

**3. AI / Inverse Design Layer**
* **Inverse Design:** Utilizes the adjoint method and gradient descent to find optimal device structures or MZI phase settings based on target unitary matrices.
* Features `optax` integration for straightforward parameter optimization.

**4. Documentation & System Design**
* See `docs/system_design.md` for a comprehensive overview of the platform's distributed architecture.

## Planning & Roadmap

### Phase 1: JAX-Native Engine Foundation
- [x] Set up immutable state data structures for the Yee Grid.
- [x] Implement 1D/2D Maxwell curl operators using jax.numpy.
- [x] Implement pure-functional time-stepping loop (jax.lax.scan).
- [x] Integrate basic sources (Dipole, Gaussian) and DFT monitors.

### Phase 2: Differentiability & The Adjoint Method
- [x] Implement density-based material parameterization (permittivity mapping).
- [x] Build custom VJP logic to bypass memory constraints of unrolling massive time steps.
- [x] Integrate Optax for basic gradient descent on a simple transmission loss function.

### Phase 3: Inverse Design of Optical Components
- [x] Extract S-parameters from FDTD for MZI mesh integration.
- [x] Optimize a basic directional coupler.
- [ ] Synthesize low-loss waveguide crossings.
- [ ] Design Mach-Zehnder Interferometer (MZI) phases for active modulation.

### Phase 4: Optical Neural Network Integration
- [ ] Construct MVM (Matrix-Vector Multiplication) layers using optimized components.
- [ ] End-to-end integration with neural network frameworks (Flax/PyTorch).
