# Perspective Projection

In the camera coordinate system, real-world measurements (e.g., in millimeters) must be converted to pixel units for image coordinates. This conversion uses scaling factors $m_x$ and $m_y$, which translate sensor measurements to pixels:

$$
m_x = \frac{\text{Number of pixels in x-axis}}{\text{Sensor width in mm}}, \quad
m_y = \frac{\text{Number of pixels in y-axis}}{\text{Sensor height in mm}}
$$

The **principal point** $(o_x, o_y)$ is where the camera's optical axis intersects the image plane, typically near the center. The **focal length**, converted to pixel units using $m_x$ and $m_y$, scales the 3D coordinates based on their depth ($Z$-coordinate):

$$
x' = m_x\frac{f \cdot X}{Z} + o_x, \quad
y' = m_y\frac{f \cdot Y}{Z} + o_y
$$

To express this projection using matrix operations, we use **homogeneous coordinates**. The **intrinsic matrix** $K$ encapsulates the camera's internal parameters:

$$
\begin{pmatrix}
x \\
y \\
w
\end{pmatrix} =
\begin{pmatrix}
f \cdot m_x & 0 & o_x & 0 \\
0 & f \cdot m_y & o_y & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix}
X \\
Y \\
Z \\
1
\end{pmatrix}
$$

The intrinsic matrix $K$ is part of the full projection matrix and is expressed as:

$$
K = \begin{pmatrix}
f \cdot m_x & 0 & o_x \\
0 & f \cdot m_y & o_y \\
0 & 0 & 1
\end{pmatrix}
$$
