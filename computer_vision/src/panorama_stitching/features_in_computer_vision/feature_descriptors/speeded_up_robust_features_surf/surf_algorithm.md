# SURF Algorithm

1. **Integral Image**: Compute the integral image for fast calculation of box-type filters, which approximate Gaussian filters.
2. **Hessian Matrix**: Use the Hessian matrix to detect keypoints. The determinant of the Hessian matrix highlights blob-like structures in the image.
3. **Keypoint Localization**: Identify local maxima and minima in the determinant of the Hessian matrix across scales.
4. **Descriptor Construction**:
   - Assign orientation based on Haar wavelet responses.
   - Divide the region around the keypoint into 4x4 subregions.
   - Compute Haar wavelet responses in each subregion to capture gradient information.
   - Normalize the descriptor to achieve robustness against lighting changes.

**SURF Properties**:

- **Speed**: Faster than SIFT due to the use of integral images and box filters.
- **Robustness**: Maintains robustness to scale, rotation, and illumination changes.
- **Efficient Matching**: Laplacian indexing enhances matching efficiency by considering the polarity of keypoints.
