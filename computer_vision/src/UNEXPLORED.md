# underexplored

## Different least squares approach in Homographies and Lukas kanade

The problem formulations for homographies and optical flow (Lukas Kanade method), are approached as least squares problems.
The method of solving these problems, however, varies due to the specifics of each mathematical model and the assumptions underlying them.
Let's explore why the approach differs and clarify the use of Singular Value Decomposition (SVD) in the context of homographies and why it might not be directly applicable or preferred in the case of optical flow:

- Homographies: In homography estimation, you typically solve for a transformation matrix that maps points from one image to another. This matrix has 8 degrees of freedom (assuming the bottom right element is normalized to 1), and the equations are set up to equate corresponding points through the transformation. The system is usually overdetermined (more equations than unknowns), as multiple point correspondences are used. SVD is employed to decompose the coefficient matrix and directly find the solution that minimizes the geometric error. The solution corresponding to the smallest singular value of the decomposed matrix is chosen because it represents the minimal residual solution—the direction in which the data is least susceptible to noise or errors. This smallest singular value effectively gives the solution vector (homography matrix elements) that best fits the overdetermined system under the constraint of minimizing the output error.
- Optical flow problem: In the Lucas-Kanade method for optical flow, you solve for the motion vectors (u, v) at each window or point in the image based on image intensity gradients. The goal is to minimize the error in brightness constancy across the window. Technically, SVD could be used in optical flow problems to solve the least squares system, similar to homographies. However, there are practical reasons this might not be the preferred approach:
  - Condition Number Sensitivity: In optical flow, especially using the Lucas-Kanade method, the focus often lies not just on finding any solution, but on ensuring that the solution is stable and reliable. If $A^TA$A is ill-conditioned (i.e., has a high condition number), using SVD without modifications may result in a solution that is overly sensitive to noise. This is because small singular values, which contribute to instability, are not regularized.
  - Regularization Considerations: Unlike in homographies where the least significant components (related to the smallest singular values) are sought, in optical flow, regularization often needs to add stability by penalizing solutions that fit the noise. Regularizing (modifying the condition number) through methods like Tikhonov regularization directly manipulates $A^TA$ to improve the conditioning before solving.

The difference boils down in a difference of goals.
Homographies focus is on exact mapping of coordinates, often under geometric transformation assumptions.
Optical Flow focus is on the flow vector field's smoothness and stability across an image sequence under dynamic conditions.

## Why in triangulation we are concerned with scale of homogeneous coordintes and other applications (homographies) not

In homographies, you're dealing with a transformation between two 2D planes (such as a planar object in 3D projected into an image), not between 3D and 2D. The matrix that defines the transformation between two planes is a homography matrix, this matrix relates the points on one plane to the points on another plane.
Homographies are defined up to scale because we are only concerned with the direction of the points (the lines in homogeneous coordinates) and not the exact magnitude.
In homographies the scale factor is inherently handled within the matrix H.
Typically to remove ambiguity the homography matrix is normalized by setting $h_{33} =1$.
This effectively fixes the scale factor and makes the transformation well-defined.

In triangulation, you're trying to estimate the 3D position of a point based on its projections onto two or more image planes.
The key difference is that triangulation involves perspective projection, where the scale factor becomes important because it represents depth (distance from the camera).

The projection matrix P introduces this scaling factor in the projection:

$$
x = \alpha P
$$

This \alpha accounts for the fact that x only gives you the direction of the projection from the camera, and the scale factor tells you how far along that direction the 3D point X lies.
In other words, \alpha represents how far away the 3D point is from the camera along the ray.

In summary:

In homographies, you're transforming between 2D planes, and the transformation itself doesn't need depth information.
You can directly normalize the matrix to remove the scale factor without needing to know \alpha.
However, in triangulation, the depth (or distance) of the 3D point is essential to reconstruct the point, so you have to explicitly account for \alpha when solving the system of equations.

## Different role of feature matching, stereo matching and triangulation

Feature matching refers to the process of identifying points of interest in multiple images that correspond to the same point in the physical world.
This step is crucial for both stereo vision and broader multi-view reconstruction processes.

Stereo matching is a specialized form of feature matching that occurs specifically in stereo vision systems, where two cameras capture images from slightly different viewpoints.
Matched points that are assumed to correspond to the same physical points in space.
This matching often involves calculating disparities (the differences in the horizontal position of corresponding features in the two images), which provide a preliminary measure of depth.

Triangulation uses the results of feature or stereo matching to calculate the actual 3D coordinates of the matched points.
o determine the precise location in 3D space of the matched features by using the geometry of the camera setup (known via the camera projection matrices and the baseline between cameras in a stereo setup).
