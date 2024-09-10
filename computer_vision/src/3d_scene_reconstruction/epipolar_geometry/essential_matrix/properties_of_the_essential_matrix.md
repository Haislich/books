# Properties of the Essential Matrix

1. **Longuet-Higgins equation**: The epipolar constraint $\mathbf{x}'^T E \mathbf{x} = 0$ must be satisfied by corresponding points in two images.
2. **Epipolar lines**: For a point $\mathbf{x}$ in one image, the corresponding epipolar line in the other image is given by:
   $$
   \mathbf{l}' = E \mathbf{x}, \quad \mathbf{l} = E^T \mathbf{x}'
   $$
3. **Epipoles**: The epipoles are the null spaces of $E$ and $E^T$, respectively:
   $$
   E \mathbf{e} = 0, \quad \mathbf{e}'^T E = 0
   $$

The essential matrix has **five degrees of freedom** (three for rotation and two for translation), and it is **rank 2**, meaning its two non-zero singular values are equal.
