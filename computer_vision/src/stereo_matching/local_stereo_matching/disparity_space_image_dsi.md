# Disparity Space Image (DSI)

The **Disparity Space Image (DSI)** represents match scores between pixel patches from two stereo images across different disparities. For each pixel in the left image, it stores the matching cost for each possible disparity.

- The DSI is used to find the disparity that minimizes the cost for each pixel.
- DSI can be visualized as a matrix where lower values indicate better matches.
