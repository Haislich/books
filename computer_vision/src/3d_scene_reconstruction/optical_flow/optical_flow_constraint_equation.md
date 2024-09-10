# Optical Flow Constraint Equation

When estimating optical flow, you generally work with two consecutive frames from a video sequence or two images taken at slightly different times, $t$ and $t + \Delta t$. The goal is to compute the motion between these frames—specifically, how each pixel or feature in the first frame moves to become a corresponding pixel or feature in the second frame.

These frames should be close enough in time to ensure minimal change in the scene other than the motion of interest.
This temporal closeness ensures that any movement between them is small and manageable. Another assumption is that the scene has stable lighting and no drastic environmental changes aside from object or camera movement.

In this setting, we make the following key assumptions:

- **Brightness Constancy Assumption**:  
  It is assumed that the brightness of any given point in the scene remains constant between the two frames. This means that if a point moves from one location to another between frames, its intensity does not change.
  $$
  I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)
  $$

- **Small Motion Assumption**:  
  Points in the image do not move far between frames, allowing for simpler mathematical treatment and avoiding large, complex displacements. This assumption permits the use of a first-order Taylor series to approximate changes, simplifying the problem to linear terms.

- **Spatial Coherence Assumption**:  
  The motion of a pixel is assumed to be similar to that of its immediate neighbors. This assumption helps define smooth motion across the image and is critical in resolving ambiguities in areas where brightness constancy alone may be insufficient.
