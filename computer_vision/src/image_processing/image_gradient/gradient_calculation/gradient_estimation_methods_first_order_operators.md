# Gradient Estimation Methods (First-Order Operators)

- **Sobel Operator**: Utilizes separate convolution kernels for detecting horizontal (\(G_x\)) and vertical (\(G_y\)) changes.
- **Prewitt Operator**: Similar to the Sobel operator but with different kernels that don't emphasize pixels directly adjacent to the central pixel.
- **Scharr Operator**: Offers optimized rotation invariance and better derivative approximation.
- **Roberts Operator**: Employs a 2x2 kernel pair, effective for detecting diagonal edges.
