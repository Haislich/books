# Affine Transformations

Affine transformations include all linear transformations plus translation. They maintain parallelism of lines and can be expressed using homogeneous coordinates in matrix form:

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} =
\begin{bmatrix}
a & b & t_x \\
c & d & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$
