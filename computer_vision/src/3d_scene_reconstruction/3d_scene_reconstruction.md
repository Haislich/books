# 3D Scene Reconstruction

3D scene reconstruction is the process of capturing the exact shape and appearance of real-world environments and objects in three dimensions using sequences of images or video.
This task involves converting observations from multiple two-dimensional images into a single three-dimensional model of a scene.

A typical 3D reconstruction pipeline involves:

- **Optical Flow**: Provides a basis for understanding the apparent motion of objects in the image sequence. It can be a preliminary step in estimating how objects move relative to the camera. Optical flow helps differentiate between static backgrounds and moving objects in the scene. By understanding these differences, it's possible to segment the scene more accurately and reconstruct moving and static parts separately.

- **Camera Calibration**: Essential for determining the intrinsic parameters of the camera (such as focal length, optical center, and lens distortion). This step ensures that the 3D reconstruction is scaled and oriented correctly relative to the real world.

- **Epipolar Geometry**: Involves understanding the geometric relationship between multiple views of a scene taken from different camera positions. This is crucial for reducing errors in feature matching between images and simplifying algorithms by reducing the search space to 1D epipolar lines instead of 2D images.

- **Stereo Matching**: Uses epipolar geometry to find corresponding points between pairs of images taken from slightly different viewpoints. By identifying these correspondences, it’s possible to compute depth information via triangulation, leading to a dense 3D reconstruction of the scene.
