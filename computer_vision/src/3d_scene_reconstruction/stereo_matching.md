# Stereo Matching

Stereo matching involves finding corresponding points between two stereo images taken from slightly different viewpoints. The goal is to compute the **disparity** for each pair of corresponding points, which is the difference in their horizontal positions in the two images.

The cameras are typically aligned so that their imaging planes are parallel, and corresponding points lie on the same horizontal lines (epipolar lines). The **baseline** $b$ is the known distance between the two camera centers.
