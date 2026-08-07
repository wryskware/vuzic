# Research: WebGPU Simulation Substrates & the Latent→Parameter Mapping

Surveyed 2026-08. Goal: a large, fluid, stateful, deterministic
compute-shader simulation whose RULES are modulated by a latent timeline, with
recurrence in music producing recognizably related visual states.

## Comparison table

| System | Visual | Memory/state | Param robustness | WebGPU cost | Existing code |
|---|---|---|---|---|---|
| Lenia | organic creatures | strong but brittle | **very poor** (thin fractal shell) | med | none mature |
| **Flow-Lenia** | flowing protoplasm | **excellent** (mass + advected param field) | **good** (mass conservation forbids blowup/death) | med-high | none (JAX only) |
| Particle Lenia | crisp cells/rotors | good, multistable | good | low | tiny |
| Neural CA | learned/specific | strong | n/a (weights, needs training) | low | W