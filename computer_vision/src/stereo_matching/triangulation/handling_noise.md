# Handling Noise

In an ideal scenario without noise, the rays projected from different cameras through corresponding points should intersect exactly at the location of the 3D point. However, due to noise in the data (e.g., pixel quantization, camera calibration errors), these rays often do not intersect perfectly.

Since the rays do not intersect perfectly due to noise, the **least squares method** is used to find the best point that minimizes the error in terms of the distance from all rays. This method computes an optimal solution that is closest to satisfying all the equations given by the projection matrices.

Mathematically, this involves setting up a system of equations derived from each camera's projection equation and solving it by minimizing the sum of squared differences (errors) between the observed projections and the projections predicted by the model.
