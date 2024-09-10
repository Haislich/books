# Local Stereo Matching

Local stereo matching computes disparity by comparing local image patches. The steps are:

1. **Disparity Range**: Set a range of possible disparities.
2. **Block Matching**: Compare blocks of pixels in the left and right images over the disparity range.
3. **Cost Calculation**: Compute similarity using metrics like **Sum of Absolute Differences (SAD)**.
4. **Disparity Selection**: Choose the disparity with the lowest cost for each pixel.

This method is fast but may struggle in low-texture areas or repetitive patterns.
