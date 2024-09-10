# Blob Detector

A **blob** refers to a region in the image that differs in properties, such as brightness or color, from surrounding areas. The **Laplacian of Gaussian (LoG)** is often used as a blob detector:

1. Apply the Gaussian filter to smooth the image.
2. Use the Laplacian filter to detect regions of rapid intensity change (blobs).
3. Identify **zero crossings** in the Laplacian response, which indicate blob boundaries.

For **scale-invariance**, this process can be applied using a **Gaussian pyramid**:

- At each level of the pyramid, apply the LoG filter to detect blobs at different scales.
- Significant blobs are those that appear consistently across scales.

This method ensures that blob-like features are detected at appropriate scales, making it a powerful tool for detecting regions of interest in an image.
