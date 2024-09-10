# Coordinate Transformation

Transforming world coordinates to camera coordinates involves the **extrinsic parameters**, which describe the camera's orientation and position in the world:

- **Rotation matrix (R)**: A 3x3 matrix representing the camera's orientation.
- **Translation vector (t)**: A vector representing the camera's position.

The transformation from world coordinates $P_W$ to camera coordinates $P_C$ is:

$$
P_C = R(P_W - t)
$$

In homogeneous coordinates, this becomes:

$$
\mathbf{E} = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$
