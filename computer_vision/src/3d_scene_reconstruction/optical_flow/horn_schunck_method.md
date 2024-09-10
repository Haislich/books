# Horn-Schunck Method

The Horn-Schunck method estimates optical flow by modeling the image as a function of continuous variables $(x, y, t)$ and the flow fields $u$ and $v$ as continuous functions of $(x, y)$. The objective is to minimize the following energy functional:

$$
E(u,v) = \int \int \underbrace{(I(x+u(x,y), y+v(x,y), t+1) - I(x,y,t))^2}_\text{quadratic penalty for brightness change} +
\lambda \underbrace{(|| \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2) dxdy}_\text{quadratic penalty for flow smoothness}
$$

- The **brightness constancy term** ensures that the intensity of a point in the image remains constant over time, enforcing the assumption that pixels maintain their appearance as they move.
- The **smoothness term** enforces smoothness in the flow field by penalizing large gradients in the flow vectors $u$ and $v$. This regularizes the flow, discouraging abrupt changes between neighboring pixels.

The parameter $\lambda$ controls the trade-off between the brightness constancy and smoothness terms:

- A larger $\lambda$ enforces more smoothness but can oversmooth motion at object boundaries.
- A smaller $\lambda$ allows more detailed motion but can produce noisy flow fields.
