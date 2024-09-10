# Template Matching

Template matching for stereo vision finds the correspondence between a small window or patch of pixels in one image and a patch in the other image of a stereo pair. The main steps are:

- A template is defined in one image (usually the left image), and the algorithm searches for the most similar template along the corresponding epipolar line in the other image.
- The similarity between the template and candidate patches in the right image is measured using metrics like **Normalized Cross-Correlation (NCC)** or **Sum of Squared Differences (SSD)**.
- The best match corresponds to the position with the highest (for NCC) or lowest (for SSD) similarity score.
- The disparity for each pixel is the horizontal difference between the matched patches, which can be related to depth based on camera geometry and baseline.
