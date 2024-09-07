# Image gradient

An image gradient is a directional change in the intensity or color in an image.
It measures how the image intensity changes between adjacent pixels and is used to find edges and other significant transitions in images.

The gradient of an image is calculated by taking derivatives in the horizontal and vertical directions.
This is usually done by convolving the image with derivative filters (kernels).

The gradient consists of two components:

- Horizontal Gradient (\[G_x\]): Measures changes in intensity in the horizontal direction.
- Vertical Gradient (\[G_y\]): Measures changes in intensity in the vertical direction

Partial derivatives represent the rate of change of image intensities with respect to the \[x\] and \[y\] directions independently.
They help in understanding the variation of pixel values along each dimension of the image.

Gradient magnitude is a measure of the strength of the gradient at each pixel.
It gives the rate at which the pixel values change at that point, and thus is indicative of the presence of edges or boundaries.
High values typically indicate strong edges where the intensity changes abruptly, while low values might suggest flat areas with little to no change in intensity.

$$
Magnitude (G) = sqrt(G_x^2 + G_y^2)
$$

Gradient Orientation indicates the direction of the greatest rate of increase of intensity from one pixel to another.
It points perpendicular to the edge direction.
Provides the angle at which these changes occur, which can be crucial for algorithms that need to understand the structure or layout of features within an image, such as in texture analysis or object recognition.

$$
Orientation (θ) = atan2(G_y, G_x)
$$

Typically, partial derivatives are computed using convolution with derivative kernels such as the Sobel, Prewitt, or Scharr operators, which are designed to emphasize changes in pixel values along one dimension while averaging out changes in the perpendicular dimension.

## Gradient estimation methods

- Sobel: The Sobel operator is one of the most widely used methods for gradient calculation in image processing.
It uses two separate convolution kernels, one for detecting horizontal changes (Gx) and one for vertical changes (Gy).
- The Prewitt operator is similar to the Sobel operator but uses different convolution kernels that do not emphasize the pixels directly adjacent to the central pixel.
-Scharr: The Scharr operator offers an alternative that optimizes the rotation invariance in gradient calculation. Its coefficients provide a better approximation to derivative computation.
-Robers: The Roberts Cross operator calculates the gradient using a pair of 2x2 convolution kernels. It is particularly effective at highlighting diagonal edges.

## Edges

Edges can be characterized as the points in an image at which the image brightness changes abruptly.
These changes usually represent the boundaries of objects, texture changes, or other significant variations in the image scene.

Edges are important features in visual data because they often define the boundaries of objects within an image, making them crucial for various tasks in image processing and computer vision, such as segmentation and object recognition.

Image gradients are directly related to edge detection because they highlight areas of the image with significant intensity changes, which correspond to edges:

- Gradient Magnitude: High gradient magnitudes indicate strong intensity changes and are therefore likely locations of edges. Wherever the gradient magnitude is high, there is likely an edge or a boundary.

- Gradient Orientation: The orientation of the gradient at each pixel points in the direction where the intensity increases most rapidly and is perpendicular to the direction of the edge. Thus, knowing the gradient orientation can help in tracing the direction or alignment of edges within an image.
