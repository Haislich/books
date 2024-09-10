# 8-Point Algorithm for Fundamental Matrix

The **eight-point algorithm** is a method to compute the fundamental matrix using at least 8 point correspondences between two images.

Steps:

1. **Normalize the points**: Translate and scale the points so that the centroid is at the origin and the average distance to the origin is $\sqrt{2}$.
2. **Set up the system of equations**: Each point correspondence provides one linear equation:
   $$
   x'_m x_m f_1 + x'_m y_m f_2 + x'_m f_3 + y'_m x_m f_4 + y'_m y_m f_5 + y'_m f_6 + x_m f_7 + y_m f_8 + f_9 = 0
   $$
3. **Assemble the matrix $A$**: Using the point correspondences, form the matrix $A$.
4. **Solve using SVD**: Compute the SVD of $A$ and take the smallest singular value.
5. **Enforce the rank-2 constraint**: Modify $F$ by setting the smallest singular value to 0.
6. **Unnormalize**: Transform $F$ back to the original scale.

The result is the fundamental matrix $F$ that best satisfies the epipolar constraint for the given point correspondences.
