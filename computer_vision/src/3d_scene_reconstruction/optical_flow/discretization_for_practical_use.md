# Discretization for Practical Use

To apply the Horn-Schunck method in practice, the continuous equation is discretized, turning the integrals into sums over the pixel grid of the image. The discretized objective function is:

$$
E(U, V) = \sum_{x, y} \left( (I_x(x, y) U(x, y) + I_y(x, y) V(x, y) + I_t(x, y))^2 \right) +
\lambda \sum_{x, y} \left( (U(x, y) - U(x+1, y))^2 + (U(x, y) - U(x, y+1))^2 + (V(x, y) - V(x+1, y))^2 + (V(x, y) - V(x, y+1))^2 \right)
$$

The resulting system of equations is large but sparse, involving only local neighborhoods of pixels. An iterative approach is commonly used to solve this system.
