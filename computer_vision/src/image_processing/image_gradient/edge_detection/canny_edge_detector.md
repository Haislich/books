# Canny Edge Detector

The Canny edge detector is one of the most effective algorithms for edge detection, adhering to Canny's criteria. The steps of the Canny algorithm are:

1. **Noise Reduction**: The image is smoothed using a Gaussian filter to reduce noise, which prevents false edges.
2. **Gradient Calculation**: Gradients are calculated, typically using the Sobel operator, to estimate horizontal (\(G_x\)) and vertical (\(G_y\)) derivatives.
3. **Non-Maximum Suppression**: Potential edges are thinned by retaining local maxima in the gradient direction, ensuring that edges remain sharp.
4. **Double Thresholding**: Two thresholds (low and high) are applied:
   - **Strong Edges**: Pixels with gradient magnitudes above the high threshold are considered strong edges.
   - **Weak Edges**: Pixels with gradient magnitudes between the thresholds are retained only if connected to strong edges.
5. **Edge Tracking by Hysteresis**: Starting from strong edges, weak edges connected to them are also marked as edges, ensuring edge continuity.
