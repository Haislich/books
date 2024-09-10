# Gaussian Pyramid

A Gaussian pyramid is a series of progressively smaller images derived from the original image. Each subsequent image is smoothed using a Gaussian filter and then sub-sampled. This process is repeated multiple times, generating a stack of images where each level is a reduced-resolution version of the previous one. The steps involved are:

1. **Apply Gaussian Blur**: Smooth the image using a Gaussian filter.
2. **Sub-sample**: Reduce the resolution of the image, typically by removing every alternate row and column.
