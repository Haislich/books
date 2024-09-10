# SIFT Algorithm

1. **Scale-Space Construction**: Construct a scale space by applying Gaussian filters at different scales. The Difference of Gaussians (DoG) is used to detect keypoints.
2. **Keypoint Localization**: Find local maxima and minima in the DoG. Keypoints are refined using Taylor expansion to achieve sub-pixel accuracy.
3. **Orientation Assignment**: Compute the gradient magnitude and orientation around each keypoint. The orientation histogram is created, and the keypoint is assigned a dominant orientation.
4. **Descriptor Computation**:
   - Calculate gradients in a 16x16 region around the keypoint.
   - Rotate the region according to the keypoint’s orientation.
   - Divide the region into 4x4 cells and compute orientation histograms for each.
   - Concatenate the histograms into a 128-element vector.

**SIFT Properties**:

- **Scale and Rotation Invariance**: SIFT features are invariant to changes in scale and orientation.
- **Robustness to Illumination**: The use of gradients makes SIFT somewhat invariant to lighting changes.
- **Distinctiveness**: The 128-element descriptor captures significant local structure, making it highly distinctive.
