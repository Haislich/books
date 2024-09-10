# Homographies

Homographies (projective transformations) extend affine transformations by allowing changes in perspective. They can warp, tilt, and scale image content and are particularly useful for stitching images with different viewpoints. Homographies are represented by a 3x3 matrix that transforms points in homogeneous coordinates:

$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix}=
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$
