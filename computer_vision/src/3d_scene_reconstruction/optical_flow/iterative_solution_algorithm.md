# Iterative Solution Algorithm

The algorithm for solving the Horn-Schunck optical flow problem works as follows:

1. **Precompute image gradients**: Compute $I_x$, $I_y$, and the temporal gradient $I_t$ from the input images.
2. **Initialize flow field**: Set initial guesses for the flow fields, typically $(u, v) = (0, 0)$.
3. **Iterative updates**: For each pixel, update the flow fields $u$ and $v$ iteratively based on neighboring pixel values.
4. **Repeat until convergence**: Continue updating the flow fields until convergence, typically when changes between iterations fall below a predefined threshold.
