# Essential Matrix

In stereo vision, the **epipolar constraint** defines the relationship between corresponding points in two views using the essential matrix. Given two corresponding points $\mathbf{x}$ in the first image and $\mathbf{x}'$ in the second image, the epipolar constraint is:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$

For a 3D point $\mathbf{X}$ observed in two cameras, the transformation between the two camera views can be expressed as:

$$
\mathbf{x}' = R (\mathbf{x} - \mathbf{t})
$$

Where $R$ is the rotation matrix and $\mathbf{t}$ is the translation vector between the two camera views.

Using the coplanarity condition, the essential matrix $E$ is derived as:

$$
E = R [\mathbf{t}]_{\times}
$$

Where $[\mathbf{t}]_{\times}$ is the skew-symmetric matrix of the translation vector $\mathbf{t}$:

$$
[\mathbf{t}]_{\times} = \begin{bmatrix} 0 & -t_3 & t_2 \\ t_3 & 0 & -t_1 \\ -t_2 & t_1 & 0 \end{bmatrix}
$$

Thus, the epipolar constraint becomes:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$
