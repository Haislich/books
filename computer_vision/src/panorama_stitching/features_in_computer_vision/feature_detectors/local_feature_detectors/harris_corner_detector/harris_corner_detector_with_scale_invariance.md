# Harris Corner Detector with Scale-Invariance

To make the Harris Corner Detector **scale-invariant**, it can be combined with the **Laplacian of Gaussian (LoG)** filter. This allows detection of features at multiple scales:

1. Apply a Gaussian filter with varying \( \sigma \) values.
2. Compute image gradients and the second moment matrix at each scale.
3. Compute the Harris response at each scale.
4. Perform non-maximum suppression across both spatial and scale dimensions to ensure that detected corners are local maxima.

This approach ensures corners are detected robustly across different scales, adapting to the intrinsic size of the features.
