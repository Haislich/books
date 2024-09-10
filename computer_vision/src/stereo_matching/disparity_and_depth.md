# Disparity and Depth

Disparity ($d$) is defined as the horizontal shift between corresponding points in the left ($u_l$) and right ($u_r$) images:

$$
d = u_l - u_r
$$

The **depth** $Z$ of a point is related to disparity, the baseline $b$, and the camera focal length $f$ by:

$$
Z = \frac{f \cdot b}{d}
$$

- **Greater disparity** indicates a closer object.
- **Lesser disparity** indicates a farther object.
