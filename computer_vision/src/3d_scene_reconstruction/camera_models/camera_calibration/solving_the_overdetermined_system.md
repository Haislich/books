# Solving the Overdetermined System

In practice, the system $\mathbf{A}\mathbf{p} = 0$ is overdetermined because we have many point correspondences but only 12 unknowns in $\mathbf{p}$. Overdetermined systems have no exact solution due to:

- **Measurement noise**: Real-world data introduces errors.
- **Imperfections**: Inaccuracies in image measurements and object geometry.

To find the best projection matrix $\mathbf{P}$, we solve the system using the **least squares method**, which minimizes the squared errors between the observed and projected points:

$$
\min || \mathbf{A} \mathbf{p} ||^2
$$
