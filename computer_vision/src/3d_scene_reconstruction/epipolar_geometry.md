# Epipolar Geometry

Epipolar geometry describes the geometric relationship between two views of a 3D scene, observed by two cameras. It's crucial in stereo vision, as it simplifies the task of finding corresponding points between images for 3D reconstruction, motion estimation, and object recognition.

The **fundamental matrix (F)** is a 3x3 matrix that encapsulates epipolar geometry between two images. Given a point in one image, the corresponding epipolar line in the other image can be computed using the fundamental matrix. It is central to uncalibrated stereo vision, where the camera intrinsics are unknown.

For **calibrated systems**, where the camera intrinsics are known, the **essential matrix (E)** encodes the relative rotation and translation (extrinsics) between two camera views. It provides a more constrained basis for estimating the relative pose between cameras.
