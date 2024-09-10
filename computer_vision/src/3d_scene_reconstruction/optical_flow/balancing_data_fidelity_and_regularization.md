# Balancing Data Fidelity and Regularization

The Horn-Schunck method aims to balance two objectives:

- **Data Fidelity**: Ensure that the computed flow respects the observed image intensities (brightness constancy).
- **Regularization**: Maintain smoothness in the flow field to avoid unrealistic discontinuities (flow smoothness).

However, minimizing the energy functional is challenging due to its non-convex nature, which means finding the global minimum can be difficult.
