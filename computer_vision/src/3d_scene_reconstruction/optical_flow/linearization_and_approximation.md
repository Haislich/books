# Linearization and Approximation

To simplify the minimization process, the brightness constancy term can be linearized using a first-order Taylor series expansion:

$$
I(x+u(x,y), y+v(x,y), t+1) \approx I(x, y, t) + I_x u(x,y) + I_y v(x,y) + I_t
$$

Where $I_x$, $I_y$, and $I_t$ are the spatial and temporal gradients of the image intensity. Substituting this into the energy functional gives:

$$
E(u,v) = \int \int \left( I_x(x,y) u(x,y) + I_y(x,y) v(x,y) + I_t(x,y) \right)^2 + \lambda \left( || \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2 \right) dxdy
$$
