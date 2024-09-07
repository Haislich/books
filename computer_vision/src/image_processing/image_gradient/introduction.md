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

## Gradient estimation methods (First order operators)

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

### Noise consideration on edges

Noise can indeed cause significant issues in edge detection by creating false positives or obscuring real edges.
Thus, preprocessing steps like applying a Gaussian filter (to smooth the image and reduce noise) before calculating the gradient (as in the Laplacian of Gaussian filter) are essential.
This step helps to ensure that the edges detected are more likely to be true transitions in image content rather than artifacts of noise.

### Laplacian filter

The Laplacian filter is a second-order derivative filter used in image processing to detect areas of rapid intensity change, which are typically associated with edges.
Unlike first-order methods (like the Sobel operator, which detects edges by measuring the gradient of image intensity), the Laplacian filter measures the rate of change of the gradients themselves, offering a different perspective on edge detection.

#### Derivative theorem of convolution

The derivative theorem of convolution is a fundamental concept in signal processing that also applies to image processing.
It states that the derivative of a convolution of two functions is the convolution of one function with the derivative of the other function.
Mathematically, it can be expressed as:

$$
(f * g)' = f' * g = f * g'
$$

#### Laplacian of gaussian

The Laplacian of Gaussian combines two distinct image processing techniques: Gaussian smoothing and the Laplacian operator.
It's particularly effective at detecting edges in images by highlighting regions of rapid intensity change.

First, a Gaussian filter is applied to the image.
This step is crucial as it helps to reduce noise and minor fluctuations in image intensity, which can lead to false edges.

The Laplacian operator is then applied to the smoothed image.
The Laplacian is a second-order derivative measure used to find areas of rapid change in images.
Because it is a second-order derivative, the sign of the Laplacian can indicate whether an edge is a transition from light to dark (negative to positive) or dark to light (positive to negative).
Zero Crossing refers to points in the image where the Laplacian of the Gaussian changes sign, indicating potential edges.
In practical terms, a zero crossing occurs where the second derivative (Laplacian) of the image intensity passes through zero.
These points are likely locations of edges because they signify a change from a maximum to a minimum or vice versa in the image gradient, typically surrounding a region where the image intensity has a steep slope.

### Edge detection

An "optimal edge detector" in image processing is a theoretical concept that aims to accurately identify the true edges in an image while minimizing false detections and noise effects.
John Canny defined the criteria for an optimal edge detector in his seminal paper on edge detection (Canny, 1986).
According to Canny, an optimal edge detector should have the following properties:

- Good Detection: The algorithm should mark as many real edges in the image as possible.
- Good Localization: Edges detected by the algorithm should be as close as possible to the true edges.
- Minimal Response: The edge detector should only mark a point as an edge once. In other words, the detector should minimize the number of local maxima around the true edge.

#### Canny edge detector

The Canny edge detector is widely regarded as one of the most effective edge detection algorithms and is designed to be an optimal edge detector by adhering closely to these criteria.
The steps involved in the Canny edge detection algorithm are:

1. Noise Reduction:The image is first smoothed by a Gaussian filter to reduce noise and details that could lead to false edges. This helps in achieving the minimal response criterion by eliminating spurious gradients in the image.
2. Gradient Calculation: The gradients of the smoothed image are then calculated, typically using the Sobel operator to estimate the derivatives in both the horizontal (Gx) and vertical (Gy) directions.
3. Non-Maximum Suppression:This step thins the potential edges to ensure that the detected edges are sharp. For each pixel, it checks if it is a local maximum in the direction of the gradient (i.e., the pixel's gradient magnitude is larger than the magnitudes in the positive and negative gradient directions).
Double Thresholding:
4. Double Thresholding: To further reduce false edges, a double thresholding strategy is used. It defines two thresholds, a low and a high threshold:
    - Strong Edges: Pixels with gradient magnitudes above the high threshold are considered strong edge pixels.
    - Weak Edges: Pixels with gradient magnitudes between the low and high thresholds are considered weak edge pixels and are only retained if they are connected to strong edge pixels.
5.Edge Tracking by Hysteresis: Finally, this step involves tracking along the edges using hysteresis. Starting from strong edges, the weak edges that are connected to strong edges are included as part of the edges. This helps in preserving edge continuity and ensuring good detection and localization.
