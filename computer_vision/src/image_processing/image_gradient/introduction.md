# Image Gradient

An image gradient represents the directional change in intensity or color within an image. It measures how the image intensity changes between adjacent pixels and is often used to detect edges and other significant transitions.

## Gradient Calculation

The gradient of an image is computed by taking derivatives in both the horizontal and vertical directions. Typically, this is done by convolving the image with derivative filters (kernels). The gradient consists of two components:

- **Horizontal Gradient** (\(G_x\)): Measures changes in intensity in the horizontal direction.
- **Vertical Gradient** (\(G_y\)): Measures changes in intensity in the vertical direction.

Partial derivatives represent the rate of change of image intensities with respect to the \(x\) and \(y\) directions independently. These derivatives provide insight into how pixel values vary across each dimension of the image.

### Gradient Magnitude

The magnitude of the gradient at a pixel measures the strength of intensity changes, indicating the presence of edges or boundaries. High gradient magnitudes typically signal strong edges where intensity changes abruptly, while low values suggest flatter regions.

\[
\text{Magnitude} (G) = \sqrt{G_x^2 + G_y^2}
\]

### Gradient Orientation

The orientation of the gradient reveals the direction of the greatest rate of increase in intensity from one pixel to another. It is perpendicular to the edge direction and provides the angle at which intensity changes occur, aiding in feature detection, such as in texture analysis or object recognition.

\[
\text{Orientation} (\theta) = \text{atan2}(G_y, G_x)
\]

### Gradient Estimation Methods (First-Order Operators)

- **Sobel Operator**: Utilizes separate convolution kernels for detecting horizontal (\(G_x\)) and vertical (\(G_y\)) changes.
- **Prewitt Operator**: Similar to the Sobel operator but with different kernels that don't emphasize pixels directly adjacent to the central pixel.
- **Scharr Operator**: Offers optimized rotation invariance and better derivative approximation.
- **Roberts Operator**: Employs a 2x2 kernel pair, effective for detecting diagonal edges.

## Edges

Edges are points where the image brightness changes abruptly, often indicating the boundaries of objects or significant variations in the scene. They are critical in image processing for tasks like segmentation and object recognition. Image gradients are directly related to edge detection since they emphasize regions with notable intensity changes.

- **Gradient Magnitude**: High magnitudes indicate strong intensity changes, likely marking the location of edges.
- **Gradient Orientation**: Points in the direction of the steepest intensity increase and is perpendicular to the edge direction.

### Noise Considerations in Edge Detection

Noise can create false edges or obscure true edges. Preprocessing with a Gaussian filter to reduce noise, as used in the Laplacian of Gaussian (LoG) method, is often necessary to improve edge detection accuracy.

## Laplacian Filter

The Laplacian filter is a second-order derivative filter that detects areas of rapid intensity change. Unlike first-order methods (such as Sobel), which measure the gradient of intensity, the Laplacian measures the rate of change of the gradients, offering a different perspective on edge detection.

### Derivative Theorem of Convolution

The derivative theorem of convolution is a key concept in image processing. It states that the derivative of a convolution is the convolution of one function with the derivative of the other:

\[
(f *g)' = f'* g = f * g'
\]

### Laplacian of Gaussian

The Laplacian of Gaussian (LoG) combines Gaussian smoothing with the Laplacian operator. It is effective for detecting edges by highlighting areas of rapid intensity change.

1. **Gaussian Smoothing**: First, a Gaussian filter is applied to reduce noise.
2. **Laplacian Operation**: The Laplacian is applied to the smoothed image to detect areas of intensity change. The sign of the Laplacian helps identify whether an edge is a transition from light to dark or vice versa.

### Zero Crossing

A zero crossing occurs where the Laplacian changes sign, indicating potential edges. This typically corresponds to regions where image intensity exhibits a steep slope, making it a useful method for edge detection.

## Edge Detection

An "optimal edge detector" aims to accurately identify true edges while minimizing false detections and the effects of noise. According to John Canny's criteria (Canny, 1986), an optimal edge detector should have:

- **Good Detection**: Detect as many true edges as possible.
- **Good Localization**: Ensure detected edges are close to the actual edges.
- **Minimal Response**: Avoid multiple responses to a single edge.

### Canny Edge Detector

The Canny edge detector is one of the most effective algorithms for edge detection, adhering to Canny's criteria. The steps of the Canny algorithm are:

1. **Noise Reduction**: The image is smoothed using a Gaussian filter to reduce noise, which prevents false edges.
2. **Gradient Calculation**: Gradients are calculated, typically using the Sobel operator, to estimate horizontal (\(G_x\)) and vertical (\(G_y\)) derivatives.
3. **Non-Maximum Suppression**: Potential edges are thinned by retaining local maxima in the gradient direction, ensuring that edges remain sharp.
4. **Double Thresholding**: Two thresholds (low and high) are applied:
   - **Strong Edges**: Pixels with gradient magnitudes above the high threshold are considered strong edges.
   - **Weak Edges**: Pixels with gradient magnitudes between the thresholds are retained only if connected to strong edges.
5. **Edge Tracking by Hysteresis**: Starting from strong edges, weak edges connected to them are also marked as edges, ensuring edge continuity.
