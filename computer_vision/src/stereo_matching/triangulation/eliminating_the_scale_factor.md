# Eliminating the Scale Factor

To eliminate the scale factor, we use a geometrical trick: the cross product of two collinear vectors is zero:

$$
\mathbf{x} \times (\mathbf{P} \mathbf{X}) = 0
$$

Assuming $\mathbf{x} = [x \ y \ 1]^T$, the cross product of $\mathbf{x}$ and $\mathbf{P}\mathbf{X}$ expands as:

$$
\mathbf{x} \times \mathbf{P} \mathbf{X} = \begin{bmatrix}
y \mathbf{p}_3^T \mathbf{X} - \mathbf{p}_2^T \mathbf{X} \\
\mathbf{p}_1^T \mathbf{X} - x \mathbf{p}_3^T \mathbf{X} \\
x \mathbf{p}_2^T \mathbf{X} - y \mathbf{p}_1^T \mathbf{X}
\end{bmatrix} = \mathbf{0}
$$

Where $\mathbf{p}_1^T$, $\mathbf{p}_2^T$, and $\mathbf{p}_3^T$ are the rows of the projection matrix $\mathbf{P}$. This gives us three equations, but due to homogeneity, only two are independent.
