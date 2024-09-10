# Triangulation

Triangulation is a crucial concept in stereo vision used to determine the three-dimensional coordinates of a point from its projections in two (or more) images taken from different viewpoints. It involves identifying the 3D point that corresponds to a specific pair of 2D points observed from different camera positions.

Given a set of (noisy) matched points $\{\mathbf{x}_i, \mathbf{x}_i'\}$ and camera matrices $\{\mathbf{P}, \mathbf{P}'\}$, the goal is to estimate the 3D point $\mathbf{X}$.
