# Forward Imaging Model

The forward imaging model describes how 3D points in the world are projected onto a 2D image sensor using a set of mathematical transformations. It accounts for both **intrinsic** parameters (e.g., focal length, principal point, and distortion) and **extrinsic** parameters (position and orientation of the camera) to accurately map 3D coordinates to 2D image coordinates.

The forward imaging model consists of two main steps:

1. **Coordinate transformation**: Transforms world coordinates (3D points in space) into camera coordinates.
2. **Perspective projection**: Projects the camera coordinates onto the 2D image plane.

Unlike the idealized pinhole camera model, the forward imaging model corrects for real-world effects like lens distortion and imperfections, providing a more accurate depiction of how cameras capture images.
