# Projection Equation

With known projection matrices, the projection equation is:

$$
\mathbf{x} = \mathbf{P} \mathbf{X}
$$

In real-world scenarios, we observe the scaled 2D coordinates $\mathbf{x}$ (not homogeneous coordinates). The relationship between the 2D projection and the 3D point can be expressed with a scale factor $\alpha$:

$$
\mathbf{x} = \alpha \mathbf{P} \mathbf{X}
$$

Here, $\alpha$ is the unknown scale factor relating the 2D coordinates to the homogeneous projection. This reflects that $\mathbf{x}$ and $\mathbf{P} \mathbf{X}$ are collinear vectors.
