# Common Blending Techniques

- **Feathering**:
  - **Weighted Mask**: A gradient mask is applied across the overlap, gradually blending the images. Pixels near the edge of the overlap have lower weight, while those toward the center have higher weight.
  - **Linear Interpolation**: The pixel values in the overlap are computed by linearly interpolating between corresponding pixel values, weighted by the mask. This reduces visible seams but may cause ghosting if the images aren’t perfectly aligned.

- **Pyramid Blending**:
  - **Image Pyramids**: Each image is decomposed into a multi-scale pyramid (e.g., using Gaussian pyramids) with progressively lower resolutions.
  - **Blend Each Layer**: Corresponding layers from each pyramid are blended separately.
  - **Reconstruct**: The final image is reconstructed from the blended pyramid layers. This method handles variations across different scales, reducing ghosting and improving smooth transitions.

- **Laplacian Pyramid Blending**:
  - **Laplacian Pyramids**: These pyramids capture image details by subtracting each Gaussian pyramid level from the next, isolating finer details at each scale.
  - **Blend and Reconstruct**: The Laplacian pyramids are blended, and the final image is reconstructed from the combined pyramid. This technique excels at preserving edge details, making it ideal for images with complex overlaps.
