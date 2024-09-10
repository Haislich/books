# Lucas-Kanade Method

The Lucas-Kanade method assumes that the optical flow is essentially constant in a local neighborhood around each pixel. Instead of solving for the flow at each pixel independently, it finds a single flow vector that best fits all pixels within a window centered around the target pixel.

For each pixel in the window, the optical flow constraint equation is:

$$
I_x(x_i, y_i) u + I_y(x_i, y_i) v = -I_t(x_i, y_i)
$$

Where:

- $I_x(x_i, y_i)$ and $I_y(x_i, y_i)$ are the spatial image gradients,
- $I_t(x_i, y_i)$ is the temporal image gradient,
- $u$ and $v$ are the optical flow components.

Since $u$ and $v$ are assumed to be constant across the window, this leads to a system of equations for all pixels in the window, which can be written in matrix form as:

$$
\begin{bmatrix}
I_x(x_1, y_1) & I_y(x_1, y_1) \\
I_x(x_2, y_2) & I_y(x_2, y_2) \\
\vdots & \vdots \\
I_x(x_n, y_n) & I_y(x_n, y_n)
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix} =
\begin{bmatrix}
-I_t(x_1, y_1) \\
-I_t(x_2, y_2) \\
\vdots \\
-I_t(x_n, y_n)
\end{bmatrix}
$$

Denoting the matrix of spatial gradients as $A$, the vector of temporal gradients as $\mathbf{b}$, and the flow vector as $\mathbf{u} = [u, v]^T$, the system can be written compactly as:

$$
A \mathbf{u} = \mathbf{b}
$$

The system is typically overdetermined, meaning there are more equations than unknowns, which makes it impossible to satisfy all equations exactly. Instead, the Lucas-Kanade method uses the least squares approach to find an optimal solution that minimizes the error across all equations:

$$
\mathbf{u} = (A^T A)^{-1} A^T \mathbf{b}
$$
