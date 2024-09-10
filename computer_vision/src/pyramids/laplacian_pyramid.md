# Laplacian Pyramid

A Laplacian pyramid is closely related to the Gaussian pyramid and is formed by taking the difference between consecutive images in the Gaussian pyramid. It captures the image detail lost between one level of the Gaussian pyramid and the next finer level. The process involves:

1. **Create Gaussian Pyramid**: First, create the Gaussian pyramid.
2. **Form Laplacian Layers**: Each layer of the Laplacian pyramid is formed by subtracting the expanded version of the upper Gaussian layer from the corresponding Gaussian layer.

The Laplacian pyramid is particularly useful in image compression and enhancement because it stores detailed information, making it easier to reconstruct finer details during the expansion process.
