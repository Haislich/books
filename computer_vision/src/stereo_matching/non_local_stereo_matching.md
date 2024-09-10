# Non-Local Stereo Matching

Non-local algorithms incorporate more extensive spatial contexts or even global image information to determine disparity. These methods often use:

- **Dynamic Programming**: Incorporates a smoothness constraint along a scanline to find an optimal match by minimizing a global cost function.
- **Graph Cuts**: Models the problem as a graph where each pixel is a node, and disparity choices are modeled as edges. The goal is to find a cut that minimizes the total disparity error across the image.
