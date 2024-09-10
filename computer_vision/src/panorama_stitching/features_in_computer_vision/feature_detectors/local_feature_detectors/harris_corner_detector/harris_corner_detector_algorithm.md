# Harris Corner Detector Algorithm

1. Compute image gradients \( I_x \) and \( I_y \) (often using a Sobel filter), and smooth them with a Gaussian filter:
    $$
    S_{xx} = G * I_x^2, \quad S_{yy} = G * I_y^2, \quad S_{xy} = G * I_x I_y
    $$
2. Construct the second moment matrix \( A \) using the smoothed gradients:
    $$
    \mathbf{A} = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}
    $$
3. Compute the **corner response function**:
    $$
    R = \det(\mathbf{A}) - k \cdot (\text{trace}(\mathbf{A}))^2
    $$
    Where:
    $$
    \det(\mathbf{A}) = S_{xx} S_{yy} - S_{xy}^2, \quad \text{trace}(\mathbf{A}) = S_{xx} + S_{yy}
    $$

4. Apply a threshold to \( R \) to detect potential corners and use **non-maximum suppression** to keep only local maxima.
