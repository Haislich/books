# Small Motion Assumption

Under the small motion assumption, the intensity change due to a small shift \( \Delta u \) is approximated linearly using a Taylor expansion:

$$
I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u
$$

Substituting into the SSD gives us:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(\nabla I(x_i) \cdot \Delta u)^2]
$$

This can be further expanded:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(I_x(x_i) u_x + I_y(x_i) u_y)^2]
$$

This leads to the quadratic form:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) \begin{bmatrix} u_x & u_y \end{bmatrix} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix}
$$

Defining the matrix \( \mathbf{A} \) as:

$$
\mathbf{A} = \sum_{i} w(x_i) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

Matrix \( A \), also known as the **structure tensor**, captures the local image gradients, which help in determining whether a region is a corner, edge, or flat area. The eigenvalues of \( A \) describe the local image structure:

- Large, comparable eigenvalues indicate a corner.
- One large eigenvalue suggests an edge.
- Small eigenvalues correspond to flat regions.
