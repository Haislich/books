# Optical Flow vs Motion Field

In the context of optical flow and computer vision, the **motion field** refers to a 2D vector field that represents the projection of the actual 3D motion of points within a scene onto the image plane. This field indicates the real motion paths that points in the observed scene follow from one frame to another, due to either the movement of the camera, the objects, or both.

The motion field can arise from various sources, including the relative motion between the camera and the scene objects (e.g., a camera passing by stationary objects or rotating around a fixed point), or the independent motion of objects within the scene (e.g., cars moving on a road).

It geometrically represents how each point in three-dimensional space moves between frames in terms of two-dimensional vectors mapped onto the camera's image plane. The motion field attempts to capture real-world movement, as opposed to **optical flow**, which only captures *apparent* motion—how the motion appears to an observer, which can be influenced by factors like lighting changes, reflections, and other visual artifacts.

While the true motion field represents the actual physical movement of objects in 3D space, this information is typically inaccessible from a single viewpoint without additional data like depth cues or multiple camera views. Instead, what we can compute directly from image sequences is the optical flow, which represents the apparent motion in the 2D image plane.

Optical flow and the motion field ideally represent the same phenomenon—the movement of objects and features in a scene. However, optical flow, which is derived from changes in image brightness, does not always accurately reflect the actual physical motion described by the motion field.
