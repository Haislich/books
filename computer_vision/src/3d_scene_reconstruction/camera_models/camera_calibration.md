# Camera Calibration

Camera calibration is the process of determining the intrinsic and extrinsic parameters of a specific camera so that it aligns with a mathematical model. Calibration uses real-world images of known patterns (e.g., checkerboards) to compute the camera’s parameters, including lens distortions, ensuring accurate 3D-to-2D transformations.

The calibration process adjusts the mathematical model to fit the camera's physical characteristics, accounting for imperfections such as lens distortion and sensor misalignment.

For each 3D world point $P_W$ and 2D image point $p$, the projection equation is:

$$
u = \frac{p_{11} X + p_{12} Y + p_{13} Z + p_{14}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}, \quad
v = \frac{p_{21} X + p_{22} Y + p_{23} Z + p_{24}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}
$$

Multiplying both sides by the denominator to avoid division gives:

$$
u \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{11} X + p_{12} Y + p_{13} Z + p_{14}
$$
$$
v \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{21} X + p_{22} Y + p_{23} Z + p_{24}
$$

These equations can be represented as a system for each point correspondence:

$$
\mathbf{A} \mathbf{p} = 0
$$
