# Pyramids

In computer vision, pyramids are a concept used to represent an image at different scales, typically to enable multi-scale processing and analysis.
There are mainly two types of pyramids: Gaussian pyramids and Laplacian pyramids.
Both types help in various applications like image compression, feature extraction, image blending, and more efficient and effective image analysis.

## Gaussian Pyramid

A Gaussian pyramid is a series of progressively smaller images derived from the original image, where each subsequent image is smoothed using a Gaussian filter and then sub-sampled. This process is repeated multiple times, generating a stack of images where each level is a reduced-resolution version of the previous one. The steps involved are:

1. Apply Gaussian Blur: Smooth the image using a Gaussian filter.
2. Sub-sample: Reduce the resolution of the image, typically by removing every alternate row and column.

## Laplacian Pyramid

A Laplacian pyramid is closely related to the Gaussian pyramid and is formed by the difference between images in the Gaussian pyramid.
It captures the image detail lost between one level of the Gaussian pyramid and the next finer level.
The process involves:

1. Create Gaussian Pyramid: First, create the Gaussian pyramid.
2. Form Laplacian Layers: Each layer of the Laplacian pyramid is formed by subtracting the expanded version of the upper Gaussian layer from the corresponding Gaussian layer.
The Laplacian pyramid is particularly useful in image compression and enhancement because it stores detailed information in the image, making it easier to reconstruct the finer details during the expansion process.

The Laplacian pyramid allows for a perfect reconstruction of the original image.
By storing the difference between each level of the Gaussian pyramid and its upsampled version, the Laplacian pyramid effectively keeps track of the exact data lost during each downsampling step.
When reconstructing the image, these differences are added back, step-by-step, to the upsampled images, allowing for an exact reconstruction of the original image.

## Steerable

## Wavelets
