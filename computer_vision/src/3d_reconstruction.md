# 3D Scene Reconstruction

3D scene reconstruction is the process of capturing the exact shape and appearance of real-world environments and objects in three dimensions using sequences of images or video.
This task involves converting observations from multiple two-dimensional images into a single three-dimensional model of a scene.

A typical 3D reconstruction pipeline involves:

- **Optical Flow**: Provides a basis for understanding the apparent motion of objects in the image sequence. It can be a preliminary step in estimating how objects move relative to the camera. Optical flow helps differentiate between static backgrounds and moving objects in the scene. By understanding these differences, it's possible to segment the scene more accurately and reconstruct moving and static parts separately.

- **Camera Calibration**: Essential for determining the intrinsic parameters of the camera (such as focal length, optical center, and lens distortion). This step ensures that the 3D reconstruction is scaled and oriented correctly relative to the real world.

- **Epipolar Geometry**: Involves understanding the geometric relationship between multiple views of a scene taken from different camera positions. This is crucial for reducing errors in feature matching between images and simplifying algorithms by reducing the search space to 1D epipolar lines instead of 2D images.

- **Stereo Matching**: Uses epipolar geometry to find corresponding points between pairs of images taken from slightly different viewpoints. By identifying these correspondences, it’s possible to compute depth information via triangulation, leading to a dense 3D reconstruction of the scene.

## Optical Flow

Optical flow is a concept in computer vision that represents the pattern of apparent motion of objects, surfaces, and edges in a visual scene, as perceived through variations in image brightness across frames.
This apparent motion arises when there is relative movement between an observer (such as a camera) and the scene being observed.

In essence, optical flow captures the displacement of pixels between consecutive video frames, representing this motion as a vector field, where each vector indicates the direction and speed of motion at that point in the image.

Optical flow allows for the analysis of the dynamics within a scene
It is crucial for understanding how objects or features move over time, which is important for tasks such as object tracking and predicting future positions.

By analyzing the motion of points in a scene, optical flow can also contribute to the 3D reconstruction of scene geometry.

### Optical Flow vs Motion Field

In the context of optical flow and computer vision, the **motion field** refers to a 2D vector field that represents the projection of the actual 3D motion of points within a scene onto the image plane. This field indicates the real motion paths that points in the observed scene follow from one frame to another, due to either the movement of the camera, the objects, or both.

The motion field can arise from various sources, including the relative motion between the camera and the scene objects (e.g., a camera passing by stationary objects or rotating around a fixed point), or the independent motion of objects within the scene (e.g., cars moving on a road).

It geometrically represents how each point in three-dimensional space moves between frames in terms of two-dimensional vectors mapped onto the camera's image plane. The motion field attempts to capture real-world movement, as opposed to **optical flow**, which only captures *apparent* motion—how the motion appears to an observer, which can be influenced by factors like lighting changes, reflections, and other visual artifacts.

While the true motion field represents the actual physical movement of objects in 3D space, this information is typically inaccessible from a single viewpoint without additional data like depth cues or multiple camera views. Instead, what we can compute directly from image sequences is the optical flow, which represents the apparent motion in the 2D image plane.

Optical flow and the motion field ideally represent the same phenomenon—the movement of objects and features in a scene. However, optical flow, which is derived from changes in image brightness, does not always accurately reflect the actual physical motion described by the motion field.

#### Examples of Discrepancies Between Optical Flow and Motion Field

1. **Lambertian Motion Sphere**:  
   Imagine a perfectly smooth, Lambertian sphere (which reflects light diffusely) rotating in space. The sphere's surface points are moving, hence there is a real motion field. However, if the sphere is uniformly colored and the lighting is even, there might be no change in brightness patterns detectable by an observer. Therefore, no optical flow would be observed, even though there is an actual motion field.

2. **Moving Light Around a Stationary Ball**:  
   If the ball itself is stationary, there is no actual motion of the ball's surface points, and thus, the motion field is null. However, as the light moves, it creates changing shadows and highlights on the ball's surface. These changes in brightness are captured as optical flow, indicating apparent motion where there is no actual physical movement of the object.

3. **Barber Pole Illusion**:  
   The actual motion of the stripes on a barber pole is horizontal as the pole rotates around its axis. Visually, due to the cylindrical shape and the observer's usual frontal perspective, the stripes appear to move vertically. This creates an optical flow that is perpendicular to the actual direction of the motion field.

### Optical Flow Constraint Equation

When estimating optical flow, you generally work with two consecutive frames from a video sequence or two images taken at slightly different times, $t$ and $t + \Delta t$. The goal is to compute the motion between these frames—specifically, how each pixel or feature in the first frame moves to become a corresponding pixel or feature in the second frame.

These frames should be close enough in time to ensure minimal change in the scene other than the motion of interest.
This temporal closeness ensures that any movement between them is small and manageable. Another assumption is that the scene has stable lighting and no drastic environmental changes aside from object or camera movement.

In this setting, we make the following key assumptions:

- **Brightness Constancy Assumption**:  
  It is assumed that the brightness of any given point in the scene remains constant between the two frames. This means that if a point moves from one location to another between frames, its intensity does not change.
  $$
  I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)
  $$

- **Small Motion Assumption**:  
  Points in the image do not move far between frames, allowing for simpler mathematical treatment and avoiding large, complex displacements. This assumption permits the use of a first-order Taylor series to approximate changes, simplifying the problem to linear terms.

- **Spatial Coherence Assumption**:  
  The motion of a pixel is assumed to be similar to that of its immediate neighbors. This assumption helps define smooth motion across the image and is critical in resolving ambiguities in areas where brightness constancy alone may be insufficient.

#### Derivation of the Optical Flow Constraint Equation

Using the brightness constancy assumption and considering small motion, the image intensity function at the new location $(x + \Delta x, y + \Delta y)$ at time $(t + \Delta t)$ can be approximated using a Taylor series expansion:

$$
I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t) + I_x \Delta x + I_y \Delta y + I_t \Delta t
$$

Where:

- $I_x = \frac{\partial I}{\partial x}$ is the image gradient in the $x$ direction,
- $I_y = \frac{\partial I}{\partial y}$ is the image gradient in the $y$ direction,
- $I_t = \frac{\partial I}{\partial t}$ is the temporal image gradient.

By setting this approximation equal to the intensity at the original point under the brightness constancy assumption and rearranging, we obtain the **optical flow constraint equation**:

$$
I_x u + I_y v + I_t = 0
$$

Where $u = \frac{\Delta x}{\Delta t}$ and $v = \frac{\Delta y}{\Delta t}$ are the components of the optical flow velocity vector at a point.

#### Interpretation of the Optical Flow Constraint Equation

This equation geometrically represents a line in the $u$-$v$ plane. Every point (velocity vector) that lies on this line is a potential solution to the optical flow constraint at a given pixel. Since we have only one equation in two unknowns ($u$ and $v$), there are infinitely many solutions that satisfy the equation for each pixel.

The true motion vector could be any point on this line. To determine the exact optical flow vector, additional information or constraints, such as smoothness assumptions or multiple viewpoints, are required.

#### Aperture Problem

The aperture problem arises when motion information is available only within a limited field of view, which is common in scenarios where the camera (or aperture) captures only a small part of a larger scene.

This problem highlights a fundamental ambiguity in motion perception:

- **Limited Visibility**:  
  When viewing motion through a small aperture (literally or figuratively, such as a small window on a larger scene), it becomes challenging to discern the true direction of motion if the visible structure does not contain sufficient variation.

- **Edge Motion**:  
  For instance, if you can only see a straight edge moving, without additional context or texture, you can only detect motion along the direction parallel to the edge. Motion perpendicular to the edge becomes indiscernible because the edge appears the same regardless of its movement along its length.

Each instance of the optical flow equation provides only one constraint for the two unknown components of the motion vector (horizontal and vertical). This leads to multiple possible solutions for the true motion vector.

The optical flow calculation at any point depends on the local gradient of image brightness. In areas where this gradient is unidirectional (such as along an edge), the flow component perpendicular to this gradient remains undetermined, manifesting the aperture problem.

#### Lucas-Kanade Method

The Lucas-Kanade method assumes that the optical flow is essentially constant in a local neighborhood around each pixel. Instead of solving for the flow at each pixel independently, it finds a single flow vector that best fits all pixels within a window centered around the target pixel.

For each pixel in the window, the optical flow constraint equation is:

$$
I_x(x_i, y_i) u + I_y(x_i, y_i) v = -I_t(x_i, y_i)
$$

Where:

- $I_x(x_i, y_i)$ and $I_y(x_i, y_i)$ are the spatial image gradients,
- $I_t(x_i, y_i)$ is the temporal image gradient,
- $u$ and $v$ are the optical flow components.

Since $u$ and $v$ are assumed to be constant across the window, this leads to a system of equations for all pixels in the window, which can be written in matrix form as:

$$
\begin{bmatrix}
I_x(x_1, y_1) & I_y(x_1, y_1) \\
I_x(x_2, y_2) & I_y(x_2, y_2) \\
\vdots & \vdots \\
I_x(x_n, y_n) & I_y(x_n, y_n)
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix} =
\begin{bmatrix}
-I_t(x_1, y_1) \\
-I_t(x_2, y_2) \\
\vdots \\
-I_t(x_n, y_n)
\end{bmatrix}
$$

Denoting the matrix of spatial gradients as $A$, the vector of temporal gradients as $\mathbf{b}$, and the flow vector as $\mathbf{u} = [u, v]^T$, the system can be written compactly as:

$$
A \mathbf{u} = \mathbf{b}
$$

The system is typically overdetermined, meaning there are more equations than unknowns, which makes it impossible to satisfy all equations exactly. Instead, the Lucas-Kanade method uses the least squares approach to find an optimal solution that minimizes the error across all equations:

$$
\mathbf{u} = (A^T A)^{-1} A^T \mathbf{b}
$$

##### Conditions for a Unique Solution

A unique solution exists if $A^T A$ is invertible. The matrix $A^T A$ is known as the **second moment matrix** (or autocorrelation matrix in some contexts), and it measures the spread or variance of the image gradients within the window. This matrix is symmetric and positive semi-definite.

The eigenvalues of $A^T A$ provide crucial information about the conditioning of the system. In the context of the Lucas-Kanade method:

- When both eigenvalues are large and approximately equal, the gradients vary significantly in multiple directions. This suggests a well-conditioned system, allowing for reliable estimation of both $u$ and $v$.
- If one eigenvalue is much smaller than the other, it indicates that the gradients are primarily in one direction. This leads to the **aperture problem**, where motion orthogonal to the dominant gradient cannot be reliably determined.

In cases where both eigenvalues are small, it implies insufficient variation in gradients, making the flow estimation unreliable.

##### Coarse-to-Fine Flow Estimation

The assumption of small motion is crucial for the effectiveness of traditional optical flow methods like Lucas-Kanade. However, in scenarios with fast-moving objects or large displacements between frames, this assumption can fail, making it difficult for these methods to accurately compute motion. In such cases, a **multi-scale, coarse-to-fine approach** becomes highly valuable.

The coarse-to-fine strategy, also known as the **pyramidal approach**, involves creating a pyramid of images where each level is a down-sampled version of the original images, progressively reducing the resolution. By starting at the coarsest level (smallest image), the method first estimates the optical flow at this low resolution, where the apparent motion between frames is significantly reduced due to the smaller scale.

The steps in this process are as follows:

1. **Build Image Pyramids**:  
   Both the current and next frames are processed to generate several layers of reduced-resolution images. Each layer is a down-sampled version of the previous one, forming an image pyramid.

2. **Initial Flow Estimation**:  
   Begin at the top of the pyramid (the coarsest, smallest images) and estimate the optical flow. At this level, even large motions become manageable because of the reduced image size.

3. **Refine Flow at Each Level**:  
   Use the flow estimate from the coarser level to guide the flow estimation at the next finer level down the pyramid. Typically, this refinement involves up-sampling the flow estimate from the coarser level and using it as an initial guess for the finer level.

4. **Iterate Down to the Finest Level**:  
   Continue refining the flow estimates down the pyramid until reaching the bottom, which corresponds to the original image resolution. At each level, the flow is progressively refined for increased accuracy.

At each level, the estimated flow vectors are scaled appropriately to account for the down-sampling effect. This scaling ensures that when an estimated flow is up-sampled to the next finer level, it represents equivalent motion in the higher resolution space. Optical flow equations are solved at each level using methods like Lucas-Kanade, modified for the scale, or other algorithms suited to the specific resolution.

The coarse-to-fine approach significantly improves the accuracy of optical flow estimation in cases with large motions, as it allows for incremental refinement from a manageable approximation of the motion at lower resolutions.

#### Horn-Schunck Method

The Horn-Schunck method estimates optical flow by modeling the image as a function of continuous variables $(x, y, t)$ and the flow fields $u$ and $v$ as continuous functions of $(x, y)$. The objective is to minimize the following energy functional:

$$
E(u,v) = \int \int \underbrace{(I(x+u(x,y), y+v(x,y), t+1) - I(x,y,t))^2}_\text{quadratic penalty for brightness change} +
\lambda \underbrace{(|| \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2) dxdy}_\text{quadratic penalty for flow smoothness}
$$

- The **brightness constancy term** ensures that the intensity of a point in the image remains constant over time, enforcing the assumption that pixels maintain their appearance as they move.
- The **smoothness term** enforces smoothness in the flow field by penalizing large gradients in the flow vectors $u$ and $v$. This regularizes the flow, discouraging abrupt changes between neighboring pixels.

The parameter $\lambda$ controls the trade-off between the brightness constancy and smoothness terms:

- A larger $\lambda$ enforces more smoothness but can oversmooth motion at object boundaries.
- A smaller $\lambda$ allows more detailed motion but can produce noisy flow fields.

#### Balancing Data Fidelity and Regularization

The Horn-Schunck method aims to balance two objectives:

- **Data Fidelity**: Ensure that the computed flow respects the observed image intensities (brightness constancy).
- **Regularization**: Maintain smoothness in the flow field to avoid unrealistic discontinuities (flow smoothness).

However, minimizing the energy functional is challenging due to its non-convex nature, which means finding the global minimum can be difficult.

#### Linearization and Approximation

To simplify the minimization process, the brightness constancy term can be linearized using a first-order Taylor series expansion:

$$
I(x+u(x,y), y+v(x,y), t+1) \approx I(x, y, t) + I_x u(x,y) + I_y v(x,y) + I_t
$$

Where $I_x$, $I_y$, and $I_t$ are the spatial and temporal gradients of the image intensity. Substituting this into the energy functional gives:

$$
E(u,v) = \int \int \left( I_x(x,y) u(x,y) + I_y(x,y) v(x,y) + I_t(x,y) \right)^2 + \lambda \left( || \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2 \right) dxdy
$$

#### Discretization for Practical Use

To apply the Horn-Schunck method in practice, the continuous equation is discretized, turning the integrals into sums over the pixel grid of the image. The discretized objective function is:

$$
E(U, V) = \sum_{x, y} \left( (I_x(x, y) U(x, y) + I_y(x, y) V(x, y) + I_t(x, y))^2 \right) +
\lambda \sum_{x, y} \left( (U(x, y) - U(x+1, y))^2 + (U(x, y) - U(x, y+1))^2 + (V(x, y) - V(x+1, y))^2 + (V(x, y) - V(x, y+1))^2 \right)
$$

The resulting system of equations is large but sparse, involving only local neighborhoods of pixels. An iterative approach is commonly used to solve this system.

#### Iterative Solution Algorithm

The algorithm for solving the Horn-Schunck optical flow problem works as follows:

1. **Precompute image gradients**: Compute $I_x$, $I_y$, and the temporal gradient $I_t$ from the input images.
2. **Initialize flow field**: Set initial guesses for the flow fields, typically $(u, v) = (0, 0)$.
3. **Iterative updates**: For each pixel, update the flow fields $u$ and $v$ iteratively based on neighboring pixel values.
4. **Repeat until convergence**: Continue updating the flow fields until convergence, typically when changes between iterations fall below a predefined threshold.

#### Robustness and Extensions

The traditional Horn-Schunck method tends to produce overly smooth flow fields, particularly at object boundaries, due to the regularization parameter $\lambda$. A high $\lambda$ value can obscure important motion discontinuities.

To make the method more robust, especially at discontinuities, alternative penalty functions or robust estimation techniques can be employed. The Horn-Schunck method can be framed as **Maximum A Posteriori (MAP) inference** in a **Markov Random Field (MRF)**, where the flow fields $(U, V)$ follow a probabilistic model:

$$
p(U,V) = \frac{1}{Z} \exp(-E(U,V))
$$

Here, $E(U,V)$ represents the Gibbs energy of the flow configuration, and minimizing this energy corresponds to finding the most likely flow field.

While a Gaussian model for the Gibbs distribution (associated with quadratic penalties) struggles with outliers and sharp motion boundaries, using a **Student-t distribution** introduces heavier tails and improves robustness. This allows for better handling of discontinuities while maintaining smoothness in other areas.

#### Gradient Descent for Optimization

Gradient descent is typically used to optimize the continuous flow fields $U$ and $V$. This approach allows the algorithm to adjust the flow vectors while maintaining sensitivity to the motion dynamics captured in the image sequence.

### Optical Flow Estimation with Deep Learning

#### FlowNet and FlowNet2

FlowNet is a pioneering CNN architecture for end-to-end optical flow estimation, where the entire process—from input frames to flow fields—is handled by a single model. It uses an encoder-decoder structure, with the encoder capturing abstract representations and the decoder reconstructing the flow field.

Two variants of FlowNet exist:

- **FlowNetSimple**: Stacks two frames and processes them through a single network to directly predict the optical flow.
- **FlowNetCorr**: Uses two streams for the input frames, combining them with a correlation layer to learn displacement between the images.

**FlowNet2** improves on FlowNet by stacking multiple FlowNets, each refining the flow estimate. It introduces sub-networks to better handle large displacements, enhancing accuracy over the original model.

#### PWC-Net

PWC-Net (Pyramid, Warping, Cost volume) builds on several innovations for optical flow estimation. It processes images through a pyramid of feature representations, capturing motion at multiple scales. Large displacements are handled at coarser levels, while finer details are refined at higher resolutions.

The **warping** step aligns the second image with the first using the flow estimated from coarser levels, simplifying displacement estimation. A **cost volume** is then computed to measure the similarity between the warped image and the first image, helping estimate motion.

Starting from the coarsest level, PWC-Net iteratively refines the optical flow at each level of the pyramid, with the final output being the full-resolution flow field.

### Feature Tracking

Feature tracking in computer vision involves detecting and following distinctive points (features) across a series of images or video frames. The goal is to track these points as they move through space and time, maintaining their identity across frames.

Once features are detected, described, and matched between frames, the motion of each matched feature can be tracked. This involves estimating the motion vector or transformation that aligns each feature with its match in subsequent frames.

#### Lucas-Kanade Feature Tracker

The Lucas-Kanade method estimates the motion of selected features by assuming that the optical flow of the brightness pattern in the image window remains constant over short time intervals. It calculates the displacement vector (in the x and y directions) for each feature, minimizing the appearance difference in its neighborhood between consecutive frames.

By using a multi-scale (pyramidal) approach, the tracker can handle large motions, refining these estimates at finer scales. This results in a set of flow vectors for each tracked feature, indicating its movement from one frame to the next.

## Camera Models

A camera model is a mathematical framework that describes how a camera projects a 3D scene onto a 2D image plane. It defines the relationship between 3D world coordinates and their corresponding 2D image coordinates, using both intrinsic and extrinsic parameters.

### Intrinsic Parameters

Intrinsic parameters define the internal characteristics of the camera, including:

- **Focal length (f)**: The distance between the camera sensor and the lens, affecting the field of view and magnification.
- **Principal point (c_x, c_y)**: The point on the image sensor corresponding to the camera's optical center.
- **Skew coefficient**: Describes the skewness between the x and y pixel axes, ideally zero in well-calibrated cameras.
- **Distortion coefficients**: Parameters that account for lens distortions, including radial and tangential distortions that warp the image.

### Extrinsic Parameters

Extrinsic parameters define the camera's position and orientation in the world:

- **Rotation matrix (R)**: Describes the camera's orientation relative to the world.
- **Translation vector (t)**: Defines the camera's position in the world relative to a reference point.

### Pinhole Camera Model

The pinhole camera model is a simplified and idealized representation that assumes an infinitely small aperture (pinhole) through which light rays pass to form an image on the image plane. It ignores lens effects like distortion, making it a practical model for many applications due to its simplicity.

In this model, 3D points from the scene are projected through the pinhole onto a 2D image plane, with depth relationships preserved—distant objects appear smaller than closer ones. The image is formed behind the pinhole at a distance determined by the **focal length (f)**.

Since the pinhole camera model does not account for lens distortion, it is ideal for scenarios where precise projection without optical corrections is needed.

### Forward Imaging Model

The forward imaging model describes how 3D points in the world are projected onto a 2D image sensor using a set of mathematical transformations. It accounts for both **intrinsic** parameters (e.g., focal length, principal point, and distortion) and **extrinsic** parameters (position and orientation of the camera) to accurately map 3D coordinates to 2D image coordinates.

The forward imaging model consists of two main steps:

1. **Coordinate transformation**: Transforms world coordinates (3D points in space) into camera coordinates.
2. **Perspective projection**: Projects the camera coordinates onto the 2D image plane.

Unlike the idealized pinhole camera model, the forward imaging model corrects for real-world effects like lens distortion and imperfections, providing a more accurate depiction of how cameras capture images.

#### Perspective Projection

In the camera coordinate system, real-world measurements (e.g., in millimeters) must be converted to pixel units for image coordinates. This conversion uses scaling factors $m_x$ and $m_y$, which translate sensor measurements to pixels:

$$
m_x = \frac{\text{Number of pixels in x-axis}}{\text{Sensor width in mm}}, \quad
m_y = \frac{\text{Number of pixels in y-axis}}{\text{Sensor height in mm}}
$$

The **principal point** $(o_x, o_y)$ is where the camera's optical axis intersects the image plane, typically near the center. The **focal length**, converted to pixel units using $m_x$ and $m_y$, scales the 3D coordinates based on their depth ($Z$-coordinate):

$$
x' = m_x\frac{f \cdot X}{Z} + o_x, \quad
y' = m_y\frac{f \cdot Y}{Z} + o_y
$$

To express this projection using matrix operations, we use **homogeneous coordinates**. The **intrinsic matrix** $K$ encapsulates the camera's internal parameters:

$$
\begin{pmatrix}
x \\
y \\
w
\end{pmatrix} =
\begin{pmatrix}
f \cdot m_x & 0 & o_x & 0 \\
0 & f \cdot m_y & o_y & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
\begin{pmatrix}
X \\
Y \\
Z \\
1
\end{pmatrix}
$$

The intrinsic matrix $K$ is part of the full projection matrix and is expressed as:

$$
K = \begin{pmatrix}
f \cdot m_x & 0 & o_x \\
0 & f \cdot m_y & o_y \\
0 & 0 & 1
\end{pmatrix}
$$

#### Coordinate Transformation

Transforming world coordinates to camera coordinates involves the **extrinsic parameters**, which describe the camera's orientation and position in the world:

- **Rotation matrix (R)**: A 3x3 matrix representing the camera's orientation.
- **Translation vector (t)**: A vector representing the camera's position.

The transformation from world coordinates $P_W$ to camera coordinates $P_C$ is:

$$
P_C = R(P_W - t)
$$

In homogeneous coordinates, this becomes:

$$
\mathbf{E} = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

#### Full Projection Matrix

The full projection matrix $P$ combines both intrinsic and extrinsic transformations, mapping 3D world points to 2D image coordinates:

$$
\mathbf{P} = \mathbf{K} \mathbf{E}
$$

#### Lens Distortion

Real cameras introduce **lens distortion**, causing non-linear displacements of points in the image. The two main types are:

- **Radial distortion**: Affects points based on their distance from the optical center, leading to compression or stretching near the edges.
- **Tangential distortion**: Occurs when the lens is misaligned with the sensor, causing a tilting effect.

### Camera Calibration

Camera calibration is the process of determining the intrinsic and extrinsic parameters of a specific camera so that it aligns with a mathematical model. Calibration uses real-world images of known patterns (e.g., checkerboards) to compute the camera’s parameters, including lens distortions, ensuring accurate 3D-to-2D transformations.

The calibration process adjusts the mathematical model to fit the camera's physical characteristics, accounting for imperfections such as lens distortion and sensor misalignment.

For each 3D world point $P_W$ and 2D image point $p$, the projection equation is:

$$
u = \frac{p_{11} X + p_{12} Y + p_{13} Z + p_{14}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}, \quad
v = \frac{p_{21} X + p_{22} Y + p_{23} Z + p_{24}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}
$$

Multiplying both sides by the denominator to avoid division gives:

$$
u \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{11} X + p_{12} Y + p_{13} Z + p_{14}
$$
$$
v \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{21} X + p_{22} Y + p_{23} Z + p_{24}
$$

These equations can be represented as a system for each point correspondence:

$$
\mathbf{A} \mathbf{p} = 0
$$

#### Solving the Overdetermined System

In practice, the system $\mathbf{A}\mathbf{p} = 0$ is overdetermined because we have many point correspondences but only 12 unknowns in $\mathbf{p}$. Overdetermined systems have no exact solution due to:

- **Measurement noise**: Real-world data introduces errors.
- **Imperfections**: Inaccuracies in image measurements and object geometry.

To find the best projection matrix $\mathbf{P}$, we solve the system using the **least squares method**, which minimizes the squared errors between the observed and projected points:

$$
\min || \mathbf{A} \mathbf{p} ||^2
$$

#### SVD and QR Factorization

The least squares solution is obtained using **Singular Value Decomposition (SVD)**, which decomposes $\mathbf{A}$ into its singular vectors and singular values. Once the solution is found, **QR factorization** is used to extract the intrinsic and extrinsic parameters from the projection matrix.

## Epipolar Geometry

Epipolar geometry describes the geometric relationship between two views of a 3D scene, observed by two cameras. It's crucial in stereo vision, as it simplifies the task of finding corresponding points between images for 3D reconstruction, motion estimation, and object recognition.

The **fundamental matrix (F)** is a 3x3 matrix that encapsulates epipolar geometry between two images. Given a point in one image, the corresponding epipolar line in the other image can be computed using the fundamental matrix. It is central to uncalibrated stereo vision, where the camera intrinsics are unknown.

For **calibrated systems**, where the camera intrinsics are known, the **essential matrix (E)** encodes the relative rotation and translation (extrinsics) between two camera views. It provides a more constrained basis for estimating the relative pose between cameras.

### Key Elements of Epipolar Geometry

- **Epipolar Plane**: The plane that contains the baseline (line connecting the camera centers) and a point in the scene observed by both cameras.
- **Epipole**: The intersection of the baseline with the image plane of each camera. The epipole in one camera is the projection of the other camera's center.
- **Epipolar Line**: The line where the epipolar plane intersects the image plane. It constrains the search for corresponding points to this line, reducing the problem from 2D to 1D.

### Essential Matrix

In stereo vision, the **epipolar constraint** defines the relationship between corresponding points in two views using the essential matrix. Given two corresponding points $\mathbf{x}$ in the first image and $\mathbf{x}'$ in the second image, the epipolar constraint is:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$

For a 3D point $\mathbf{X}$ observed in two cameras, the transformation between the two camera views can be expressed as:

$$
\mathbf{x}' = R (\mathbf{x} - \mathbf{t})
$$

Where $R$ is the rotation matrix and $\mathbf{t}$ is the translation vector between the two camera views.

Using the coplanarity condition, the essential matrix $E$ is derived as:

$$
E = R [\mathbf{t}]_{\times}
$$

Where $[\mathbf{t}]_{\times}$ is the skew-symmetric matrix of the translation vector $\mathbf{t}$:

$$
[\mathbf{t}]_{\times} = \begin{bmatrix} 0 & -t_3 & t_2 \\ t_3 & 0 & -t_1 \\ -t_2 & t_1 & 0 \end{bmatrix}
$$

Thus, the epipolar constraint becomes:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$

#### Properties of the Essential Matrix

1. **Longuet-Higgins equation**: The epipolar constraint $\mathbf{x}'^T E \mathbf{x} = 0$ must be satisfied by corresponding points in two images.
2. **Epipolar lines**: For a point $\mathbf{x}$ in one image, the corresponding epipolar line in the other image is given by:
   $$
   \mathbf{l}' = E \mathbf{x}, \quad \mathbf{l} = E^T \mathbf{x}'
   $$
3. **Epipoles**: The epipoles are the null spaces of $E$ and $E^T$, respectively:
   $$
   E \mathbf{e} = 0, \quad \mathbf{e}'^T E = 0
   $$

The essential matrix has **five degrees of freedom** (three for rotation and two for translation), and it is **rank 2**, meaning its two non-zero singular values are equal.

### Fundamental Matrix

The **fundamental matrix (F)** is a generalization of the essential matrix for **uncalibrated stereo systems**, where the camera intrinsics are unknown. It relates corresponding points $\mathbf{x}$ and $\mathbf{x}'$ between two images:

$$
\mathbf{x}'^T F \mathbf{x} = 0
$$

When the intrinsic camera parameters $K$ and $K'$ are known, the fundamental matrix can be derived from the essential matrix as:

$$
F = K'^{-T} E K^{-1}
$$

#### Properties of the Fundamental Matrix

1. **Epipolar constraint**: $\mathbf{x}'^T F \mathbf{x} = 0$ must hold for corresponding points in two images.
2. **Epipolar lines**: Similar to the essential matrix, the epipolar lines for corresponding points are:
   $$
   \mathbf{l}' = F \mathbf{x}, \quad \mathbf{l} = F^T \mathbf{x}'
   $$
3. **Epipoles**: The epipoles satisfy:
   $$
   F \mathbf{e} = 0, \quad \mathbf{e}'^T F = 0
   $$

The fundamental matrix has **eight degrees of freedom** (9 parameters, minus 1 for scale). Like the essential matrix, it is **rank 2**.

### 8-Point Algorithm for Fundamental Matrix

The **eight-point algorithm** is a method to compute the fundamental matrix using at least 8 point correspondences between two images.

Steps:

1. **Normalize the points**: Translate and scale the points so that the centroid is at the origin and the average distance to the origin is $\sqrt{2}$.
2. **Set up the system of equations**: Each point correspondence provides one linear equation:
   $$
   x'_m x_m f_1 + x'_m y_m f_2 + x'_m f_3 + y'_m x_m f_4 + y'_m y_m f_5 + y'_m f_6 + x_m f_7 + y_m f_8 + f_9 = 0
   $$
3. **Assemble the matrix $A$**: Using the point correspondences, form the matrix $A$.
4. **Solve using SVD**: Compute the SVD of $A$ and take the smallest singular value.
5. **Enforce the rank-2 constraint**: Modify $F$ by setting the smallest singular value to 0.
6. **Unnormalize**: Transform $F$ back to the original scale.

The result is the fundamental matrix $F$ that best satisfies the epipolar constraint for the given point correspondences.

## Stereo Matching

Stereo matching involves finding corresponding points between two stereo images taken from slightly different viewpoints. The goal is to compute the **disparity** for each pair of corresponding points, which is the difference in their horizontal positions in the two images.

The cameras are typically aligned so that their imaging planes are parallel, and corresponding points lie on the same horizontal lines (epipolar lines). The **baseline** $b$ is the known distance between the two camera centers.

### Disparity and Depth

Disparity ($d$) is defined as the horizontal shift between corresponding points in the left ($u_l$) and right ($u_r$) images:

$$
d = u_l - u_r
$$

The **depth** $Z$ of a point is related to disparity, the baseline $b$, and the camera focal length $f$ by:

$$
Z = \frac{f \cdot b}{d}
$$

- **Greater disparity** indicates a closer object.
- **Lesser disparity** indicates a farther object.

### Image Rectification

If the images are not aligned, **rectification** transforms the images to align the epipolar lines horizontally, simplifying stereo matching. Rectification involves applying homographies to warp the images such that corresponding points lie on the same horizontal lines. This is achieved by decomposing the essential matrix $E$ into rotation ($R$) and translation ($t$), then computing rectifying homographies.

### Local Stereo Matching

Local stereo matching computes disparity by comparing local image patches. The steps are:

1. **Disparity Range**: Set a range of possible disparities.
2. **Block Matching**: Compare blocks of pixels in the left and right images over the disparity range.
3. **Cost Calculation**: Compute similarity using metrics like **Sum of Absolute Differences (SAD)**.
4. **Disparity Selection**: Choose the disparity with the lowest cost for each pixel.

This method is fast but may struggle in low-texture areas or repetitive patterns.

#### Template Matching

Template matching involves comparing small patches of pixels in one image with patches in the corresponding horizontal line of the other image. The similarity is measured using metrics like **Normalized Cross-Correlation (NCC)** or **Sum of Squared Differences (SSD)**.

#### Stereo Block Matching

In block matching, small blocks around each pixel are compared across a range of disparities to find the best match. The **disparity** that minimizes the cost is selected as the correct match.

1. Set a disparity range [0, D].
2. For each block in the left image, slide it along the corresponding row in the right image.
3. Calculate the similarity score for each position.
4. Select the disparity with the best score.
5. Apply **left-right consistency** checks to improve reliability.

### Disparity Space Image (DSI)

The **Disparity Space Image (DSI)** represents match scores between patches of pixels from two stereo images across different disparities. For each pixel in the left image, it stores the matching cost for each possible disparity.

- The DSI is used to find the disparity that minimizes the cost for each pixel.
- DSI can be visualized as a matrix, where lower values indicate better matches.

### Non-local Stereo Matching

Non-local algorithms incorporate broader image contexts or global information. Examples include:

- **Dynamic Programming**: Minimizes a global cost function with smoothness constraints.
- **Graph Cuts**: Models the stereo matching problem as a graph and finds a minimum-cut solution to minimize disparity errors.

### Energy Minimization in Stereo Matching

Stereo matching can be formulated as an **energy minimization problem**. The goal is to find the disparity map $d$ that minimizes the energy function:

$$
E(d) = E_d(d) + \lambda E_s(d)
$$

Where:

- $E_d(d)$ is the **data term** that measures the similarity between corresponding pixels in the left and right images.
- $E_s(d)$ is the **smoothness term** that penalizes large disparity changes between neighboring pixels.
- $\lambda$ is a weighting factor that balances the two terms.

The data term $E_d(d)$ is often calculated using **SSD** or similar metrics:

$$
E_d(d) = \sum_{(x, y)} C(x, y, d(x, y))
$$

Where $C(x, y, d(x, y))$ is the cost of assigning disparity $d$ to pixel $(x, y)$.

The smoothness term $E_s(d)$ is given by:

$$
E_s(d) = \sum_{(p, q)} V(d_p, d_q)
$$

Where $V(d_p, d_q)$ penalizes disparity differences between neighboring pixels $p$ and $q$. Common penalties include the **L1 norm** or **Potts model**.

### Dynamic Programming in Stereo Matching

Dynamic programming efficiently computes disparities by recursively finding the minimum cost for each pixel based on its neighbors. The cost function is:

$$
D(x, y, d) = C(x, y, d) + \min_{d'} \left( D(x - 1, y, d') + \lambda |d - d'| \right)
$$

This minimizes both the data term $C(x, y, d)$ and the smoothness term, ensuring a globally optimal disparity map.

Local stereo matching algorithms calculate disparity based on the similarity of local image patches. These methods typically follow these steps:

Disparity Range: Set a range of possible disparities.
Block Matching: For each pixel, compare a block (or patch) of pixels around it in one image to corresponding blocks in the other image over the range of disparities.
Cost Calculation: Calculate a similarity score (such as Sum of Absolute Differences, SAD) for each block comparison.
Disparity Selection: Assign to each pixel the disparity that results in the lowest cost, often using a winner-takes-all strategy.
This approach is relatively simple and fast but often falls short in areas with low texture or repetitive patterns due to relying solely on local information.

#### Template matching

Template matching for stereo vision is a method to find the correspondence between a small window or patch of pixels in one image and a patch in the other image of a stereo pair.
The main steps involve defining a template in one image and searching for the most similar template along the corresponding epipolar line in the other image.

A template is a small window centered around a pixel in the left image.
The size of the window affects the robustness and accuracy of matching.

The search area is located along the horizontal epipolar line in the right image, considering the rectification has aligned these lines across both images.
The similarity between the template in the left image and each candidate patch along the epipolar line in the right image is measured. This can be quantified using metrics such as Normalized Cross-Correlation (NCC), Sum of Squared Differences (SSD), or other similarity measures.
For each pixel in the left image, the algorithm slides the template along the corresponding epipolar line in the right image.
The similarity measure is calculated for each position of the window along the line.
The position that gives the highest similarity score (for NCC) or the lowest score (for SSD) is considered the best match.
The disparity for each pixel is calculated as the difference in horizontal positions between the matched patches in the left and right images.
This disparity is directly related to the depth of the object point from the cameras, based on the camera geometry and baseline.

#### Stereo block matching

The block matching algorithm computes the disparity between two stereo images by comparing small blocks or windows of pixels from both images. The goal is to find the horizontal shift (disparity) at which the blocks from two images match best.

Define the maximum and minimum disparities to be considered, typically represented as [0,D], where D is the maximum disparity.
This range limits where the algorithm searches for matches, which is crucial for reducing computation time and avoiding implausible matches.
For example if D is set to 30, the algorithm will check for matches up to 30 pixels to the left in the right image.

After definining the range, Select a small region or block around each pixel (x,y) in the left image.
These blocks are used as the basis for matching, providing a more robust comparison than single pixels, which might be susceptible to noise and minor variations in intensity.
For each block centered around (x,y) in the left image, slide this block along the corresponding row in the right image, within the disparity range [0,D].
At each possible disparity d within this range, calculate a similarity score between the block from the left image and a block from the right image shifted by d pixels.
The disparity d that yields the best similarity score (highest for NCC, lowest for SSD) is chosen as the correct disparity for the center pixel (x,y) of the block in the left image.
For both images run a consistency check (left-right consistency), which ensures that the disparity found by comparing the left image to the right is consistent with the disparity found by comparing the right image to the left. This step helps identify and remove incorrect matches often caused by occlusions.
If the disparity for a pixel in the left image does not match the disparity calculated from the right image, it can be flagged as unreliable and potentially corrected or interpolated.
Occlusions can cause disparities to appear incorrect because a match might not exist in the other image.
Similarly, regions with low texture or repetitive patterns can yield ambiguous matches.
Techniques such as interpolating disparities in occluded regions or applying a smoothing filter across the disparity map can help mitigate these issues.

#### disparity search algorithm

DSI encapsulates the calculated match scores between patches of pixels from two rectified stereo images (typically left and right images).
Each pixel on a scanline in the left image is compared against pixels on the corresponding scanline in the right image over a range of disparities.
Think of the DSI as a matrix where one dimension represents the pixels along a scanline of the left image, and the other dimension represents possible disparities.
Each element in this matrix, C(i,j), corresponds to a match score between a patch centered around pixel i in the left image and a patch centered at pixel j in the right image where j varies from i−d to i+d within a feasible disparity range.

The primary use of the DSI is to facilitate the computation of the optimal disparity for each pixel, which in turn helps in generating a depth map of the scene. Depth maps are critical in applications such as 3D reconstruction, autonomous navigation, and virtual reality.
A simple way to use the DSI is through a greedy algorithm where, for each pixel i in the left image, the disparity j that minimizes the dissimilarity (or maximizes the similarity) is chosen. This corresponds to finding the minimum in each column of the DSI matrix if searching along one dimension.
Visually, the DSI can be displayed as an image where each column represents the disparity scores for a specific pixel in the left image across all tested disparities.
Darker values might represent lower costs (better matches), and brighter values indicate higher costs (poorer matches).

#### Non local stereo matching

Non-local algorithms, in contrast, incorporate more extensive spatial contexts or even global image information to determine disparity. They often use methods like:

Dynamic Programming: Incorporates a smoothness constraint along a scanline to find an optimal match by minimizing a global cost function.
Graph Cuts: Models the problem as a graph where each pixel is a node and disparity choices are modeled as edges. It aims to find a cut in the graph that minimizes the total disparity error across the image.

#### DSI with inter scanline onsistency

Inter-scanline consistency refers to the process of enforcing continuity and coherence in the disparity mapping across consecutive rows or scanlines in an image during stereo matching.
This consistency is crucial because disparities tend to change gradually across an image except at object boundaries or occlusions.

1. Match Scoring: Each pixel in a scanline from the left image is compared with pixels in the corresponding scanline of the right image. This is done using a similarity or dissimilarity score, such as Sum of Absolute Differences (SAD), Normalized Cross-Correlation (NCC), etc. These scores are computed for every possible disparity within a defined range and recorded in the DSI.
2. Path finding: The goal is to find a path through the DSI that minimizes the total dissimilarity score, which represents the sequence of disparities for each pixel in the scanline. Assigning disparities to all pixels in left scanline now amounts to finding a connected path through the DS IThis path should be smooth, indicating small changes in disparity between consecutive pixels, which is typically enforced by adding penalties for large jumps in disparity from one pixel to the next.
3. Dynamic programming algorithms, such as the Viterbi algorithm, are employed to efficiently find the path with the lowest cumulative cost. This algorithm considers not only the individual pixel dissimilarities but also the penalties for disparity changes. This ensures that the resulting disparity map is not only accurate in matching corresponding pixels across images but also smooth across the image, adhering to the physical reality of the scene. A left-right consistency check may be applied afterward to verify that the disparity value determined for a pixel in the left image matches the disparity obtained from the right image. This helps to detect and correct occlusions where a pixel in one image does not have a counterpart in the other.

#### Stereo matching as an energy minimization problem

Stereo matching as an energy minimization problem is a fundamental approach in computer vision, particularly for generating disparity maps from stereo images. The goal is to find the correspondence between pixels in two images (left and right) of the same scene taken from slightly different viewpoints. By modeling this as an energy minimization problem, the aim is to find the disparity for each pixel that minimizes a global energy function, which typically includes terms that balance image similarity and smoothness of the disparity map.

The energy function E(d)  used in stereo matching is a combination of a data term and a smoothness term. The total energy for a given disparity d is defined as:

$$
E(d) = E_d(d) + \lambda E_s(d)
$$

Where E_d(d) is the data term E_s(d) is the smoothness term and \lambda is the weighting parameter.

The data term measures the similarity between the corresponding pixels in the left and right images. For stereo matching, this could be calculated using Sum of Squared Differences (SSD) between the pixel intensities in corresponding windows.

$$
E_d(d) = \sum_{(x, y) \in I} C(x, y, d(x, y))
$$

Where C(x, y, d(x,y)) represents the cost of assigning disparity d(x, y) to a pixel at position (x,y) caclulated as:

$$
C(x, y, d) = \text{SSD}(\text{window centered at } (x, y) \text{ in left image, window centered at } (x + d(x, y), y) \text{ in right image})
$$

The smoothness term is generally given by:

$$
E_s(d) = \sum_{(p, q) \in \mathcal{E}} V(d_p, d_q)
$$

Where V(d_p, d_q) is a penalty function applieed to the disparity values of neighboring pixels p and q.
A common choice fo V is the L1 norm or the potts model
The dynamic programming approach aims to find disparities for all pixels such that the total energy E(d) across the image is minimized

 This is done by:

Local Decision at Each Pixel: For each pixel at position (x,y), and for each possible disparity d, the cost function D(x,y,d) is calculated.

$$
D(x, y, d) = C(x, y, d) + \min_{d'} \left\{ D(x - 1, y, d') + \lambda \left| d - d' \right| \right\}
$$
This function considers:
Immediate cost

C(x,y,d), which corresponds to the data term E_d (d), indicating the cost of assigning disparity d at that pixel.
Transition cost: This cost is derived from the disparity assigned to the neighboring pixel. Specifically, \min_{d'} \left\{ D(x - 1, y, d') + \lambda \left| d - d' \right| \right\}
 computes the minimum cost of reaching the current pixel from the previous one  (at x−1,y) with a disparity d', including a penalty for changing the disparity to d. This term encourages the preservation of the smoothness condition encapsulated by E_s(d)

by computing D(x,y,d) for each pixel and disparity, the algorithm effectively constructs a path through the image where each step (pixel-to-pixel transition) is optimized to both match the stereo images well  (low (C(x,y,d))) and maintain smooth disparity changes  (minimized (\lambda \left| d - d' \right|)) . The result is a disparity map that minimizes the overall energy E(d) across the entire image.

The recursive nature od D(x,y,d)  leverages the computed costs of previous decisions, ensuring that the cumulative energy is minimized in an efficient manner.

### Deep Models for Stereo Matching

#### Siamese Networks

Siamese networks for stereo matching use convolutional neural networks (CNNs) to extract features from stereo image pairs. These features are then correlated, often via dot product, to measure similarity between corresponding points. Disparity is determined by selecting the maximum correlation score for each pixel, with options for further refinement using global optimization techniques. This deep learning approach improves robustness in handling occlusions and textureless regions compared to traditional methods.

#### DispNet

DispNet is the first end-to-end deep neural network designed for stereo matching. Its architecture, inspired by U-Net, includes a contracting path to capture context and an expanding path for precise localization, with skip connections for better detail retention. A correlation layer processes disparities to compute similarity between patches of stereo images. Using multi-scale loss, DispNet is trained to handle varying levels of difficulty, improving accuracy across diverse stereo vision tasks.

#### Stereo Mixture Density Networks (SMD-Nets)

SMD-Nets use mixture density networks to predict high-resolution disparity maps by modeling the probability distribution of disparities for each pixel. This probabilistic approach improves handling of ambiguities and occlusions, preserving edge details and structural integrity in complex scenes. SMD-Nets provide sharper and more accurate disparity maps, particularly in challenging environments.
