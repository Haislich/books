# Laplacian of Gaussian

The Laplacian of Gaussian (LoG) combines Gaussian smoothing with the Laplacian operator. It is effective for detecting edges by highlighting areas of rapid intensity change.

1. **Gaussian Smoothing**: First, a Gaussian filter is applied to reduce noise.
2. **Laplacian Operation**: The Laplacian is applied to the smoothed image to detect areas of intensity change. The sign of the Laplacian helps identify whether an edge is a transition from light to dark or vice versa.
