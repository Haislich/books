# HoG Algorithm

HoG combines gradient, spatial structuring, and orientation normalization into a coherent descriptor. The steps to compute HoG are:

1. **Preprocessing**: Convert the image to grayscale to reduce complexity and focus on structure rather than color.
2. **Gradient Computation**: Compute the horizontal and vertical gradients for each pixel, often using a Sobel filter.
3. **Orientation Binning**: Divide the image into cells (e.g., 8x8 pixels). For each cell, create a histogram of gradient orientations, typically with 9 to 18 bins covering 0 to 180 degrees (unsigned) or 0 to 360 degrees (signed).
4. **Descriptor Blocks**: Group adjacent cells into larger blocks (e.g., 2x2 cells). Normalize histograms within each block to reduce sensitivity to lighting variations.
5. **Concatenation**: Combine the normalized histograms from all blocks into a single feature vector representing the HoG descriptor.
6. **Sliding Window**: For object detection, apply HoG within a sliding window across the image.

**HoG Applications**: HoG is particularly effective in human detection, capturing vertical and horizontal edges typical in human forms.
