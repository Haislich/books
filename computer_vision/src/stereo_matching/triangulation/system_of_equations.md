# System of Equations

For each 2D-3D point correspondence, we get two independent equations. For a single image point $\mathbf{x}$ and its corresponding 3D point $\mathbf{X}$, the system of equations looks like this:

$$
\begin{bmatrix}
y \mathbf{p}_3^T - \mathbf{p}_2^T \\
\mathbf{p}_1^T - x \mathbf{p}_3^T
\end{bmatrix} \mathbf{X} = \mathbf{0}
$$

If we have multiple cameras (or two views), we can concatenate the equations from each view to form a larger system. For example, with two cameras and two projection matrices, we get:

$$
\begin{bmatrix}
y_1 \mathbf{p}_{13}^T - \mathbf{p}_{12}^T \\
\mathbf{p}_{11}^T - x_1 \mathbf{p}_{13}^T \\
y_2 \mathbf{p}_{23}^T - \mathbf{p}_{22}^T \\
\mathbf{p}_{21}^T - x_2 \mathbf{p}_{23}^T
\end{bmatrix} \mathbf{X} = \mathbf{0}
$$

This forms a homogeneous system of equations $\mathbf{A} \mathbf{X} = 0$.
