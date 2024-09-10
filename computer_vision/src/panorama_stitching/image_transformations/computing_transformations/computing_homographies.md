# Computing Homographies

Homographies allow for perspective transformations, represented by a 3x3 matrix. The general form is:

$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix} =
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

The non-linear relationship between input and output coordinates is linearized by cross-multiplying, leading to:

$$
x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13} \\
y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}
$$

Each matched point provides two equations, requiring at least four point pairs to solve for the eight unknowns (assuming \( h_{33} = 1 \) to resolve scale ambiguity).

To minimize error, least squares finds:

$$
\|Ah - b\|^2
$$

Where \( A \) is constructed from the linearized equations, and \( h \) is the homography vector. Singular Value Decomposition (SVD) is used to solve this overconstrained system, where the optimal \( h \) is the eigenvector of \( A^T A \) corresponding to the smallest eigenvalue, minimizing the residual error.
