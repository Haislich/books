# Derivation of the Optical Flow Constraint Equation

Using the brightness constancy assumption and considering small motion, the image intensity function at the new location $(x + \Delta x, y + \Delta y)$ at time $(t + \Delta t)$ can be approximated using a Taylor series expansion:

$$
I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t) + I_x \Delta x + I_y \Delta y + I_t \Delta t
$$

Where:

- $I_x = \frac{\partial I}{\partial x}$ is the image gradient in the $x$ direction,
- $I_y = \frac{\partial I}{\partial y}$ is the image gradient in the $y$ direction,
- $I_t = \frac{\partial I}{\partial t}$ is the temporal image gradient.

By setting this approximation equal to the intensity at the original point under the brightness constancy assumption and rearranging, we obtain the **optical flow constraint equation**:

$$
I_x u + I_y v + I_t = 0
$$

Where $u = \frac{\Delta x}{\Delta t}$ and $v = \frac{\Delta y}{\Delta t}$ are the components of the optical flow velocity vector at a point.
