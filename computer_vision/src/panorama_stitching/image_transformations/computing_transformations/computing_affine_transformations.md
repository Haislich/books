# Computing Affine Transformations

For affine transformations, which include translation, rotation, scaling, and shearing, the relationship between matched points is:

$$
x'_i = a \cdot x_i + b \cdot y_i + t_x \\
y'_i = c \cdot x_i + d \cdot y_i + t_y
$$

This introduces six unknowns \( a, b, c, d, t_x, t_y \). The least squares method minimizes:

$$
\min_{a, b, c, d, t_x, t_y} \sum_{i=1}^n \left((x'_i - (a \cdot x_i + b \cdot y_i + t_x))^2 + (y'_i - (c \cdot x_i + d \cdot y_i + t_y))^2\right)
$$

At least three matched points are required to compute the affine transformation.
