# Photonics Simulation Platform: System Design

## Overview

The Photonics Simulation Platform is designed to bridge device-level physical simulation (FDTD) and circuit-level behavioral modeling (MZI mesh) into a unified, scalable EDA pipeline. By running on a distributed Kubernetes infrastructure, the platform natively supports high-throughput optimization and inverse design workloads.

## System Architecture

The architecture is divided into three primary tiers: the Application Interface, the Orchestration & Scheduling Layer, and the Execution Layer.

### 1. Application & API Layer
* **API Server (Go):** Exposes REST/gRPC endpoints for submitting simulation jobs (e.g., FDTD extraction, MZI phase optimization).
* **Job Definitions:** Users submit job parameters (geometry, target unitary matrix, material constraints).
* **Caching Layer (Redis):** Caches S-parameters for previously simulated components (e.g., identical directional couplers) to bypass expensive FDTD reruns when resolving large MZI circuits.

### 2. Orchestration & Scheduling Layer (Kubernetes)
* **Message Broker (NATS):** Distributes incoming jobs into specialized queues.
  * **Heavy Queue:** Long-running, compute-intensive FDTD simulations.
  * **Light Queue:** Fast, matrix-multiplication-based MZI circuit evaluations and inverse design steps.
* **Job Scheduler (Go):** Consumes from NATS and orchestrates the lifecycle of Kubernetes Jobs.
* **Autoscaling:** Integrates with Karpenter or Cluster Autoscaler to dynamically provision GPU nodes for heavy FDTD workloads and scale them down to zero when idle.

### 3. Execution Layer (Workers)
* **FDTD Worker Nodes (GPU/TPU):** Run the `uh-fdtd` JAX-accelerated engine. Responsible for full Maxwell equation solving and S-parameter extraction.
* **MZI Circuit & AI Worker Nodes (CPU/GPU):** Consume S-parameters and run the `optax` based inverse design pipeline. Adjusts phase settings via gradient descent to match target optical transformations.

### 4. Observability & Infrastructure
* **GitOps:** Infrastructure state is managed via Terraform and deployed using ArgoCD.
* **Metrics:** Prometheus scrapes metrics from the API server, NATS, and worker nodes (e.g., job latency, queue depth, GPU utilization).
* **Tracing:** OpenTelemetry provides distributed tracing across the pipeline, allowing developers to see the exact time spent in FDTD vs. Circuit optimization.

## Workflow: Device to Circuit Pipeline

1. **Job Submission:** A user submits a request to synthesize a target 4x4 unitary matrix using an MZI mesh.
2. **Device Resolution:** The scheduler checks the cache for the required directional couplers. If missing, an FDTD job is dispatched to a GPU worker.
3. **FDTD Execution:** The `uh-fdtd` engine simulates the coupler, calculates overlaps via DFT monitors, and extracts the 2x2 S-matrix.
4. **Data Handoff:** The S-matrix is saved to object storage (S3) and the cache is updated.
5. **Circuit Optimization:** An Inverse Design job is triggered. It loads the S-matrices, constructs the full mesh transfer matrix, and uses JAX automatic differentiation to tune the ideal phase shifters until the overall matrix matches the target unitary.
6. **Result Delivery:** The final optimized phases and anticipated transmission losses are returned to the user.

## Why this Architecture?

This architecture directly addresses the impedance mismatch between device and circuit simulations. FDTD is heavily compute-bound (Maxwell's equations), while circuit simulation is memory/data-bound (transfer matrix multiplication). By decoupling these via a robust queueing and caching system, the platform mimics the scalability of modern distributed testing orchestrators, making it ideal for the massive search spaces required in AI accelerator design.
