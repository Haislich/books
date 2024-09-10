# Properties of the Fundamental Matrix

1. **Epipolar constraint**: $\mathbf{x}'^T F \mathbf{x} = 0$ must hold for corresponding points in two images.
2. **Epipolar lines**: Similar to the essential matrix, the epipolar lines for corresponding points are:
   $$
   \mathbf{l}' = F \mathbf{x}, \quad \mathbf{l} = F^T \mathbf{x}'
   $$
3. **Epipoles**: The epipoles satisfy:
   $$
   F \mathbf{e} = 0, \quad \mathbf{e}'^T F = 0
   $$

The fundamental matrix has **eight degrees of freedom** (9 parameters, minus 1 for scale). Like the essential matrix, it is **rank 2**.
