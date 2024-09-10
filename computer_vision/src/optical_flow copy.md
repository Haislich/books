# 3D scene reconstruction

3D scene reconstruction is the process of capturing the exact shape and appearance of real-world environments and objects in three dimensions using sequences of images or video.
This task involves converting observations from multiple two-dimensional images into a single three-dimensional model of a scene.

A typical 3d reconstruction pipeline involves:

- Optical Flow: Provides a basis for understanding the apparent motion of objects in the image sequence, which can be a preliminary step in estimating how objects move relative to the camera. Optical flow can help differentiate between static background and moving objects in the scene. By understanding these differences, it's possible to segment the scene more accurately and reconstruct moving and static parts separately.
- Camera Calibration: Essential for determining the intrinsic parameters of the camera (like focal length and optical center) and its distortion characteristics. This step ensures that the 3D reconstruction is scaled and oriented correctly relative to the real world.
- Epipolar Geometry: Involves understanding the geometric relationship between multiple views of a scene taken from different points. This is crucial for error reduction in feature matching between images and for simplifying algorithms by reducing the search space to 1D lines instead of 2D images.
- Stereo Matching: Uses the epipolar geometry to find corresponding points between pairs of images taken from slightly different viewpoints. By identifying these correspondences, it’s possible to compute depth information via triangulation, leading to a dense 3D reconstruction of the scene.

## Optical Flow

Optical flow is a concept in computer vision that represents the pattern of apparent motion of objects, surfaces, and edges in a visual scene, as perceived through the variations in image brightness across frames.
This apparent motion arises when there is relative movement between an observer (such as a camera) and the scene being observed.
In essence, it captures the displacement of pixels between consecutive video frames, representing this motion as a vector field where each vector shows the direction and speed of motion at that point in the image.

Optical flow allows for the analysis of the dynamics within a scene.
It's crucial for understanding how objects or features move over time, which can be pivotal for tasks like tracking objects and predicting future positions.

By analyzing the motion of points in a scene, optical flow can contribute to the 3D reconstruction of the scene geometry.

### Optical flow vs Motion Field

In the context of optical flow and computer vision, the motion field refers to a 2D vector field that represents the projection of the actual 3D motion of points within a scene onto the image plane.
This field gives an indication of the real motion paths that points in the observed scene follow from one frame to another, due to either the movement of the camera, the objects, or both.

The motion field can arise due to various sources, including the relative motion between the camera and the scene objects (such as a camera passing by stationary objects or rotating around a fixed point), or the independent motion of objects within the scene (like cars moving on a road).

It geometrically represents how each point in a three-dimensional space moves between frames in terms of two-dimensional vectors mapped onto the camera's image plane.
The motion field tries to capture the real-world movement, as opposed to optical flow, which only captures apparent motion—how the motion appears to an observer, which can be influenced by lighting changes, reflections, and other visual artifacts.

The true motion field represents the actual physical movement of objects in 3D space, but this is typically inaccessible from a single viewpoint without additional information like depth cues or multiple camera views.
Instead, what we can compute directly from image sequences is the optical flow, which represents the apparent motion in the 2D image plane.

Optical flow and the motion field ideally represent the same phenomenon—the movement of objects and features in a scene.
However, optical flow, which is derived from changes in image brightness, does not always accurately reflect the actual physical motion described by the motion field.

Here are three examples describing the discrepancies between them:

- Lambertian motion sphere:  Imagine a perfectly smooth, Lambertian sphere (which reflects light diffusely) rotating in space. The sphere's surface points are moving, hence there is a real motion field. Despite the physical motion, if the sphere is uniformly colored and the lighting is even, there might be no change in brightness patterns detectable by an observer. Therefore, optical flow might not be observed even though there is a motion field.
- Moving Light Around a Stationary Ball: If the ball itself is stationary, there is no actual motion of the ball's surface points; hence, the motion field is null.As the light moves, it creates changing shadows and highlights on the ball’s surface. These changes in brightness are captured as optical flow, indicating apparent motion where there is no actual physical movement of the object.
- Barber pole illusion: The actual motion of the stripes on a barber pole is horizontal as the pole rotates around its axis.  Visually, due to the cylindrical shape and the observer's usual frontal perspective, the stripes appear to be moving vertically upward. This creates an optical flow that is perpendicular to the actual direction of the motion field.

### Optical flow constraint equation

When estimating optical flow, you generally work with two consecutive frames from a video sequence or two images taken at slightly different times, $t$ and $\Delta t$.
The goal is to compute the motion between these frames—that is, how every pixel or feature in the first frame moves to become a pixel or feature in the second frame.

These frames should be close enough in time to ensure minimal change in the scene other than the motion of interest.
The frame closeness ensures us that any movement between them is small and manageable.
Another assumption that we make is that the scene has stable lighting and no drastic changes in the environment other than the movements of objects or the camera.

In this setting we make some assumptions:

- Brightness Constancy Assumption:  It is assumed that the brightness of any given point in the scene remains constant between the two frames. This means if a point moves from one location to another between frames, its intensity does not change.
    $$
    I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)
    $$
- Small Motion Assumption:  Points in the image do not move far between frames, allowing for simpler mathematical treatments and avoiding large, complex displacements that could complicate the estimation.  This assumption permits the use of first-order Taylor series to approximate changes, simplifying the problem to linear terms.
- Spatial Coherence Assumption: The motion of a pixel is assumed to be similar to its immediate neighbors. This assumption helps in defining the motion smoothly across the image and is critical in resolving ambiguities in areas where the brightness constancy alone is insufficient.

Given this assumptions we can derive the optical constraint equation.
Using the brightness constancy and considering small motion, the image intensity function at the new location $(x + \Delta x, y + \Delta y)$ at time $(t + \Delta t)$ can be approximated using the taylor series:

$$
I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t) + I_x \Delta x + I_y \Delta y + I_t \Delta t
$$

Setting the Taylor expansion to equal the intensity at the original point under the brightness constancy assumption and rearranging gives:

$$
I_x u + I_y v + I_t = 0
$$

Where $u = \frac{\Delta x}{\Delta t}$ and $v = \frac{\Delta y}{\Delta t}$.

This equation geometrically represents a line in the $u-v$ plane.
Every point (velocity vector) that lies on this line is a potential solution to the optical flow constraint at a given pixel.
Since we have only one equation in two unknowns there are infinitely many solutions that satisfy the equation for each pixel.
The true motion vector could be any point on this line.
To pinpoint the exact location on this line (i.e., the correct optical flow vector), additional information or assumptions are required.

#### Aperture problem

The aperture problem arises when motion information is available only within a limited field of view, which is common in scenarios where the camera (or aperture) captures only a small part of a larger scene.

This problem highlights a fundamental ambiguity in motion perception:

- Limited Visibility: When viewing motion through a small aperture (literally or figuratively, such as a small window on a larger scene), it becomes challenging to discern the true direction of motion if the visible structure does not contain sufficient variation.
- Edge Motion: For instance, if you can only see a straight edge moving, without additional context or texture, you can only detect motion along the direction parallel to the edge. Motion perpendicular to the edge becomes indiscernible because the edge appears the same regardless of its movement along its length.

Each instance of the optical flow equation provides only one constraint for the two unknown components of the motion vector (horizontal and vertical).
This lack of sufficient independent equations for each pixel leads to multiple possible solutions.
The optical flow calculation at any point depends on the local gradient of image brightness.
In areas where this gradient is unidirectional (like along an edge), the flow component perpendicular to this gradient remains undetermined, manifesting the aperture problem.

#### Lukas-Kanade method

The Lucas-Kanade method assumes that the flow is essentially constant in a local neighborhood around each pixel.
Instead of solving for the flow at each pixel independently, it finds a single flow vector that is a best fit for all pixels within a window centered around the target pixel.

After defining a window, for each pixel in it the optical flow constraint equation is:

$$
I_x(x_i, y_i) u + I_y(x_i, y_i) v = -I_t(x_i, y_i)
$$

Since the assumption is that $u$ and $v$ iare constant accross the window, this leads toa system of equations for all pixels in the window, which can be written in matrix form as:

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

Detnoting the matrices as $A,b$ and $u= [u,v]^T$ the system can be written as $Au = b$.

The system is typically overdetermined, resulting in potentially extra constraints that can conflict with one another, making it impossible to find a solution that exactly satisfies all equations.
Instead, Lucas-Kanade uses the least squares method to find an optimal solution that minimizes the error across all equations:

$$
\mathbf{u} = (A^T A)^{-1} A^T \mathbf{b}
$$

Ideally this system has a unique solution if $A^TA$ is invertible.
$A^TA$ computes the second moment matrix (also known as the autocorrelation matrix in some contexts), which effectively measures the spread or variance of the gradient vectors.
This matrix is symmetric and positive semi-definite.
The eigenvalues of $A^TA$ provide critical information about the matrix's properties. In the Lucas-Kanade method, they indicate the dominant directions of the gradients and the "strength" or certainty of these directions.
When both eigenvalues are large and roughly equal, it suggests that the gradients in the window vary significantly in more than one direction, implying a well-conditioned system that is suitable for finding a reliable solution for $u$ and $v$.
If one of the eigenvalues is much smaller than the other, it indicates that the gradients are primarily in one direction — leading to the aperture problem where motion orthogonal to this direction cannot be reliably computed.

##### Coarse to fine flow estimation

The assumption of small motion is crucial for the effectiveness of traditional optical flow methods like the Lucas-Kanade.
When this assumption fails—such as in scenarios with fast-moving objects or large displacements between frames—these methods can struggle to accurately compute motion.
This is where a multi-scale, coarse-to-fine approach becomes highly valuable.

The coarse-to-fine strategy, also known as a pyramidal approach, involves creating a pyramid of images where each level is a down-sampled version of the original images, reducing resolution progressively.
By starting at the coarsest level (smallest image), the method first estimates the optical flow at this low resolution, where the apparent motion between frames is significantly reduced due to the smaller scale.

The steps are the following:

1. Build Image Pyramids: Both the current and the next frame are processed to generate several layers of reduced resolution images.
2. Initial Flow Estimation: Begin at the top of the pyramid (the smallest images) and estimate the optical flow. Here, even large motions become manageable because of the reduced image size.
3. Refine Flow at Each Level: Use the flow estimate from the previous level to guide the flow estimation at the next level down the pyramid. This refinement typically involves up-sampling the flow estimate from the coarser level and using it as an initial guess for the current level.
4. Iterate Down to Finest Level: Continue refining the flow estimates until you reach the bottom of the pyramid, which is the original image resolution.

At each level, the estimated flow vectors are scaled appropriately to compensate for the down-sampling effect.
This scaling ensures that when an estimated flow is up-sampled to the next finer level, it represents an equivalent motion in the higher resolution space.
Optical flow equations are solved at each level using methods appropriate for the scale, such as modified versions of Lucas-Kanade or even different algorithms better suited for different resolutions.

#### Horn-Schunck method

The Horn-Schunck method aims to estimate the optical flow by considering the image as a function of continuous variables $x,y,t$ and the flow fields $u,v$ as a continuos function of $(x,y)$.
The objective is to minimize the energy functional.

$$
E(u,v) = \int \int \underbrace{(I(x+u(x,y), y+v(x,y), t+1) - I(x,y,t))^2}_\text{quadratic penalty for brightness change} +
\lambda \underbrace{(|| \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2) dxdy}_\text{quadratic penalty for flow change}
$$

Where the brightness constancy term ensures that the intensity of a point in the image remains constant over time, penalizing deviations from this assumption.
Minimizing this term ensures that the computed flow vectors move the pixels in a way that the appearance of the image remains consistent, adhering to the brightness constancy assumption.

The smoothness term enforces smoothness in the flow field by penalizing large gradients in the flow vectors $u$ and $v$.
The parameter $\lambda$ controls the trade-off between the brightness constancy term and the smoothness term.
Minimizing this term reduces noise and avoids unrealistic motion discontinuities, resulting in a more physically plausible flow field.

The combined functional balances the two competing objectives:

- Data Fidelity: Ensuring that the motion respects the observed image intensities (brightness constancy).
- Regularization: Ensuring that the motion field is smooth and free of abrupt changes (smoothness).

Directly minimizing the energy functional is difficult because the problem is highly non-convex and has many local optima.
This means finding the global minimum is challenging due to the complex landscape of the functional.

One solution is to linearize the brightness constancy assumption, to do so we approximate it with the First-Order Multivariable Taylor Series

$$
E(u,v) = \int \int (I_x(x,y,t)u(x,y) +  I_y(x,y,t)v(x,y), I_t(x,y,t))^2 +
\lambda (|| \nabla u(x,y) ||^2 + || \nabla v(x,y) ||^2) dxdy
$$

To apply this method in practice, we need to discretize the continuous equation, turning the integrals into sums over the pixel grid of the image.
The discretized objective function is:

$$
E(U, V) = \sum_{x, y} \left( (I_x(x, y) U(x, y) + I_y(x, y) V(x, y) + I_t(x, y))^2 \right) +
\lambda \left( (U(x, y) - U(x+1, y))^2 + (U(x, y) - U(x, y+1))^2 +
(V(x, y) - V(x+1, y))^2 + (V(x, y) - V(x, y+1))^2 \right)
$$

The objective is quadratic in the flow maps $U$ and $V$.
While the quadratic form of each individual term suggests a convex structure, the overall problem of optical flow estimation can still be challenging due to the inherent properties of image data and the possibility of multiple local minima, especially in real-world scenarios with complex motions and varying image intensities.
To minimize the energy functional, we take the partial derivatives with respect to $U$ and $V$ and set them to zero, forming a system of linear equations.
This leads to a large but sparse linear system because each equation involves only a small neighborhood of pixels.
However in practice the iterative approach is used.

The algorithm works as follows:

1. Precompute image gradients $I_x,I_y$
2. Precompute temporal gradient $I_t$
3. Initialize flow field $(u,v) = (0,0)$
4. While not converged compute flow field updates for each pixel

##### Making Horn-Schunck more robust The Horn & Schunck algorithm typically produces results that are coherent and visually plausible due to its global approach

The HS method tends to produce overly smooth flow fields due to its reliance on a high regularization parameter, $\lambda$.
This parameter, when set too high, leads to oversmoothing, where crucial discontinuities at the edges of moving objects are obscured.
To counteract this, robust estimation techniques are suggested to better manage the quadratic penalty's limitations, particularly at discontinuities in the flow field.

The HS method can be conceptualized through Maximum A Posteriori (MAP) inference within a Markov Random Field (MRF), employing a probabilistic model where the probability of a flow configuration $(U,V)$ is governed by a Gibbs distribution:

$$
p(U,V) = \frac{1}{Z}exp{-E(U,V)}
$$

Here, the Gibbs energy quantifies the "cost" of flow configurations, advocating for configurations that minimize this energy—typically smoother and more consistent flow fields, except at discontinuities.

The conventional Gaussian model for the Gibbs distribution, associated with quadratic penalties, does not robustly handle outliers or sharp discontinuities at object boundaries.
These drawbacks prompt a shift towards using a Student-t distribution, which offers heavier tails and greater robustness against outliers, thus maintaining essential discontinuities.

Gradient descent is utilized for continuous inference over belief propagation due to the continuous nature of $U$ and $V$.
This approach ensures that the estimation process remains sensitive to the actual dynamics captured in successive frames, providing a more detailed and accurate depiction of motion, particularly when enhanced by robust penalties.

### Optical flow estimation with deep learning

#### FlowNet and FlowNet2

FlowNet is a pioneering convolutional neural network (CNN) architecture introduced for end-to-end optical flow estimation.
In the context of optical flow estimation, the term "end-to-end" refers to a machine learning or deep learning approach where the entire process—from raw input data (such as consecutive image frames) to the final output (optical flow fields)—is handled within a single, unified model.

FlowNet utilizes an encoder-decoder architecture.
The encoder progressively downsamples the input, capturing increasingly abstract representations at lower resolutions.
The decoder then upsamples these representations to construct a full-resolution flow field.

There are two variants of FlowNet:

- FlowNetSimple: This variant stacks two consecutive frames together and feeds them through a single, generic network. This approach directly learns the mapping from stacked frames to optical flow fields.
- FlowNetCorr: Instead of stacking frames, this variant processes the images through two separate but identical streams, which are then combined using a correlation layer that helps the network learn and compute displacement fields between the two images.

FlowNet2 is an evolution of the original FlowNet, addressing some of its limitations, particularly regarding accuracy and the ability to handle large displacements.
FlowNet2 stacks multiple FlowNets sequentially, with intermediate supervisions.
This stacking allows the network to refine the flow estimates, improving precision with each step.
FlowNet2 includes specialized sub-networks designed to handle large displacements effectively, a significant challenge in the original FlowNet architecture.

#### PWC-net

PWC-Net, which stands for Pyramid, Warping, and Cost volume.
PWC-Net is a sophisticated approach for estimating optical flow that integrates several innovative components to enhance performance and accuracy.

PWC-Net uses a pyramid of features extracted from the input images, employing convolutional layers.
This pyramid approach reduces spatial resolution at each level, allowing the network to capture motion at different scales and complexities.
By processing images at various resolutions, the network can effectively handle large displacements in the optical flow.
Warping is used to align features of the second image with the first.
This process helps to make the second image more similar to the first, thereby simplifying the problem of estimating displacement vectors.
Warping adjusts the features of the second image based on the estimated flow from a coarser level, which improves the accuracy of the flow estimation at finer levels.

After warping, PWC-Net computes a cost volume for each level of the feature pyramid.
The cost volume represents the similarity between features of the first image and the warped second image, facilitating the estimation of how much each pixel has moved between frames.
The network uses normalized cross-correlation to compute these cost volumes, enhancing its invariance to color and intensity changes.

Starting from the coarsest level of the pyramid, the network estimates the optical flow and then progressively refines this estimate at finer levels.
This multi-scale approach allows the network to capture both large and subtle motions effectively.

A CNN processes the cost volume at each level to predict the flow vectors.
These predicted flows are then used to warp the second image's features for the next level's estimation, iteratively refining the flow estimation as it moves to finer scales.

The final output is the optical flow for the lowest (finest) level of the pyramid, which corresponds to the original image resolution.

## Feature Tracking

Feature tracking in the context of computer vision refers to the method of detecting and tracking distinctive points (features) across a series of images or video frames.
The goal of feature tracking is to maintain the identity and follow the motion of these points as they move through space and time in the visual field.

Assuming that feature detection, description and matching is performed across all frames in the video the movement of each matched feature can be tracked across frames.
This typically involves estimating the motion vector or transformation that best aligns each feature with its match in the subsequent frame(s).

### Lukas-Kanade feature tracker

The motion of selected features is estimated by assuming that the apparent velocity (optical flow) of the brightness pattern in the image window remains constant over short times.
The method computes the displacement vector (in x and y directions) for each feature that minimizes the difference in appearance between its neighborhood in consecutive frames.
By operating at multiple scales, the tracker can handle larger motions by tracking at coarser scales and then refining these estimates at finer scales.
This hierarchical approach allows for more effective and accurate tracking of fast-moving objects.
The output of the process is a set of flow vectors for each tracked feature, indicating the movement of each feature from one frame to the next.

## Camera models

A camera model is a mathematical framework that describes how a camera captures a 3D scene and projects it onto a 2D image plane.
It defines the relationship between the coordinates of points in the 3D world and their corresponding coordinates in the 2D image captured by the camera.
The camera model incorporates both intrinsic parameters (which define the internal characteristics of the camera, such as focal length, principal point, and lens distortion) and extrinsic parameters (which define the position and orientation of the camera in the world).

These parameters relate to the camera’s internal characteristics and are called intrinsic:

- Focal length (f): The distance between the camera sensor and the lens, which affects the field of view and magnification of the image.
- Principal point (c_x, c_y): The point on the image sensor that corresponds to the projection of the camera's optical center.
- Skew coefficient: Describes how the x and y pixel axes are skewed relative to each other, which ideally should be zero in well-calibrated cameras.
- Distortion coefficients: These parameters account for lens distortion that warps images, typically radial and tangential distortions.

These parameters define the camera’s position and orientation in the world and are called extrinsics:

- Rotation matrix (R): Describes the camera's orientation in the world.
- Translation vector (t): Describes the camera's position in the world relative to a reference point.

### Pinhole camera model

The pinhole camera model is a simplified and idealized camera model that assumes an infinitely small aperture (pinhole) through which light rays pass to form an image on the image plane.
It does not account for lens effects like distortion but is widely used because of its simplicity and effectiveness in many practical applications.

The pinhole camera captures images by projecting 3D points from the scene through a single point (the pinhole) onto a 2D image plane.
The resulting projection preserves depth relationships, with distant objects appearing smaller than closer ones.

The geometry of the pinhole camera involves a single point of projection (the pinhole), and the image is formed on an image plane placed behind the pinhole.
The focal length f determines the distance between the pinhole and the image plane.

The pinhole camera model does not include lens distortion, making it ideal for situations where precision in projection is the goal without needing to correct for optical imperfections.

### Forward imaging model

The forward imaging model is essential in computer vision because it provides a mathematical framework for mapping the 3D world onto a 2D image
The forward imaging model is a set of mathematical equations and transformations that describe how a three-dimensional point in the world is projected through a camera's optics and onto its two-dimensional image sensor.
This model takes into account the camera’s intrinsic parameters (like focal length, principal point, and lens distortions) and extrinsic parameters (position and orientation of the camera in the world) to accurately map 3D coordinates to 2D coordinates.
The forward imaging model is fundamental in bridging the gap between the physical world and its digital representation, enabling computers to interpret and interact with their environment effectively.

The forwarding imaging describes the transformation from 3D to 2D with two subsequent projections:

- Coordinate transformation: That transforms the world coordinates (The 3D space in which the object lives) into the camera coordinates.
- Perspective projection: That transforms from the camera coordinates into the image coordinates

The forward imaging model, corrects for the limitations of the idealized pinhole camera model by incorporating real-world effects such as lens distortion, diffraction, and the impact of finite pinhole size.

<!-- Another problem that can occur is the diffraction.
Diffraction occurs when light waves encounter edges, such as the aperture edges of the camera lens.
It leads to a blurring effect, especially at small apertures, which is more prominent in compact digital cameras.

This effect is not typically modeled in the geometric projection process, but its impact can be mitigated through image deblurring algorithms or through the use of diffraction-aware calibration. -->

#### Perspective projection

In the camera coordinate system, measurements might typically be in millimeters or centimeters, reflecting the actual dimensions that a sensor can capture.
However, image coordinates are in pixels, which are discrete and have different dimensions than real-world units.
This misalignment leads to the introduction of scaling factors $m_x$ and $m_y$ which convert measurements from millimeters (or another unit) on the sensor to pixels.

$$
m_x = \frac{\text{Number of pixels in x-axis}}{\text{Sensor width in mm}} \quad
m_y = \frac{\text{Number of pixels in y-axis}}{\text{Sensor height in mm}}
$$

The principal point is where the optical axis pierces the image plane and is typically near the center of the image.
It serves as a reference point to align the optical center of the lens with the pixel grid of the image sensor.
It's generally assumed that the principal point of a camera, where the optical axis intersects the image sensor, is near the center of the sensor.
However, this assumption may not always hold true, and the actual location of the principal point can indeed be unknown, which leads to the introduction of $(o_x,o_y)$ which are the unknowns representing the principal point.

The focal length, converted to pixel units using $m_x$ and $m_y$, translates the 3D scene onto a 2D image plane, scaling the x and y coordinates based on how far the scene is from the camera (depth, Z-coordinate).

The internal geometry formed by these parameters is inherently non-linear due to the division by the depth (Z-coordinate) in the projection process:

$$
x' = m_x\frac{f \cdot X}{Z} + o_x \\
y' = m_y\frac{f \cdot Y}{Z} + o_y
$$

To handle this non-linearity and make matrix operations applicable, we use homogeneous coordinates.
This allows us to represent the perspective projection transformation as a linear matrix multiplication, with the matrix $M$, which is called intrinsic matrix:

$$
\begin{pmatrix}
x \\
y \\
w
\end{pmatrix} = \begin{pmatrix}
f \cdot m_x & 0 & o_x & 0 \\
0 & f \cdot m_y & o_y & 0 \\
0 & 0 & 1 & 0
\end{pmatrix} \begin{pmatrix}
X \\
Y \\
Z \\
1
\end{pmatrix}
$$

$M$ is such that can be thought as a concatenation of two matrixes:

$$
M = [K | 0 ]
$$

We call $K$ the calibration matrix, the matrix that encapsulates all the intrinsic parameters:

$$
K = \begin{pmatrix}
f \cdot m_x & 0 & c_x \\
0 & f \cdot m_y & c_y \\
0 & 0 & 1
\end{pmatrix}
$$

#### Coordinate transformation

The coordinate transformation from world coordinates to camera coordinates involves using the extrinsic parameters of the camera.
These parameters describe the camera's position and orientation in the world.

The extrinsic parameters typically consist of:

- Rotation matrix (R): A 3x3 matrix representing the orientation of the camera relative to some world coordinate system.
- Translation vector (t): A vector that represents the position of the camera's optical center in world coordinates.

These parameters together define how to transform a point from the world coordinate system to the camera coordinate system.
The tansformation is achieved through this relation:

$$
P_C = R(P_W -t)
$$

Where $P_C$ is the point in the camera coodinates and $P_W$ is the point in the world coordinates.
This equation effectively rotates the world around the camera to align the world axes with the camera's axes and then translates the world so that the camera appears to be at the origin of the coordinate system.
The extrinsic matrix combines the effects of rotation and translation into a single transformation matrix.
The combined extrinsic matrix is typically expressed in homogeneous coordinates to accommodate both rotation and translation in a single matrix operation.
Here's how it is structured:

$$
\mathbf{E} = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

#### Full projection matrix

To form the full projection matrix, we integrate both the intrinsic and extrinsic transformations.
The full projection matrix $P$ is obtained by multiplying the intrinsic matrix $K$ with the extrinsic matrix $E$.
This matrix multiplication results in a single matrix that can be used to transform a 3D point in world coordinates directly to a 2D point inimage coordinates using homogeneous coordinates.

$$
\mathbf{P} = \mathbf{K} \mathbf{E}
$$

#### Lens distortion

Lens distortion occurs in real cameras because lenses do not perfectly focus light rays onto the image plane.
This leads to non-linear distortions, where points in the scene are displaced from their ideal positions in the image.

There are two primary types of distortion:

- Radial distortion: This type of distortion is caused by the shape of the lens and affects points based on their distance from the optical center. It can either compress or stretch points near the edges of the image.
- Tangential distortion: This type of distortion occurs when the lens is not perfectly aligned with the sensor plane, leading to a tilting effect.

### Camera calibration

Camera calibration is the process of determining the actual parameters (intrinsic and extrinsic) of a specific camera so that it fits the mathematical model.
In calibration, real-world images of known patterns (like checkerboards) are used to compute the camera’s parameters, including any distortions in the lens.

The calibration procedure adjusts the mathematical model to match the physical characteristics of the actual camera being used.
This allows for accurate 3D-to-2D transformations, taking into account the real-world imperfections of the camera (such as lens distortion and sensor alignment).

For each 3D world point $P_W$ and 2D image point $p$ the projection equation can be written as:

$$
u = \frac{p_{11} X + p_{12} Y + p_{13} Z + p_{14}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}
v = \frac{p_{21} X + p_{22} Y + p_{23} Z + p_{24}}{p_{31} X + p_{32} Y + p_{33} Z + p_{34}}
$$

By multiplying both sides of each equation by the denominator to avoid division, we get:

$$
u \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{11} X + p_{12} Y + p_{13} Z + p_{14}
v \left( p_{31} X + p_{32} Y + p_{33} Z + p_{34} \right) = p_{21} X + p_{22} Y + p_{23} Z + p_{24}
$$

These equations can be rewritten as a system of equations for each point correspondence, leading to

$$
\mathbf{A} \mathbf{p} = 0
$$

In practice the system $Ap=0$ is overdetermined.
This happens because we have many point correspondences, but only 12 unknowns in $P$.
Overdetermined systems typically have no exact solution due to:

- Measurement noise: Real-world data points will not fit perfectly into a single linear solution.
- Imperfections: Small inaccuracies in image measurements and object geometry make an exact solution impossible

To find the best possible projection matrix P, we solve this overdetermined system using a least squares method, which minimizes the sum of the squared differences (errors) between the observed and projected points. The least squares solution finds the vector p that minimizes:

$$
\min || \mathbf{A} \mathbf{p} ||^2
$$

SVD is used to decompose the matrix A into its singular vectors and singular values.

Once the solution is found we can use QR factorization to extract entrinsic and extrinsic parameters.

## Stereo vision

Stereo vision refers to the method of using two cameras positioned at different viewpoints to capture images of the same scene.

Each point in an image corresponds to a "ray" in 3D space, originating from the camera's center of projection and passing through the point on the image sensor where the light hits.
For a single camera, while you know the direction of the ray, the exact location along this ray (the depth) where the point exists is unknown without additional information—this is the ambiguity in depth from a single viewpoint.

By analyzing the slight differences in the images from each camera, stereo vision systems mimic human binocular vision to estimate the depth and three-dimensional structure of the scene.
This process leverages the disparity between corresponding points in the two images, which arises due to the cameras' different perspectives.

The fundamental principle behind stereo vision is triangulation, where the known geometry of the two cameras and the measurable disparity of points between the images are used to reconstruct the 3D coordinates of points in the scene.
The disparity is the difference in the horizontal and/or vertical position of corresponding features in the two images.

### Calibrated stereo

In calibrated stereo vision, the cameras are pre-calibrated to determine their intrinsic parameters (like focal length and optical center) and extrinsic parameters (relative orientation and position).

#### Stereo matching

Stereo matching involves finding corresponding points between two stereo images. These images are taken from slightly different viewpoints (typically horizontally offset). The main goal is to determine the disparity for each pair of corresponding points across the two images.

These cameras are assumed to be aligned so that their imaging planes are parallel, and they typically scan along corresponding horizontal lines (epipolar lines) in the images.
The distance between the two camera centers is a known distance b, called baseline.

Disparity refers to the difference in the horizontal positions of corresponding points in the left and right images captured by the two cameras.
If a point appears in one position in the left image and shifts to another position in the right image, the horizontal shift between these positions is the disparity.

$$
d = u_l - u_r
$$

Where $u_l$ is a point in the left image and $u_r$ is the same physical point in the right image.

The baseline and disparity are directly involved in computing the depth Z of a point from the cameras.
The basic relationship connecting disparity to depth involves the baseline, the cameras' focal length f, and the disparity d.
The formula to find the depth Z of a point in the scene using these parameters is:

$$
Z = \frac{f \cdot b}{d}
$$

Disparity is crucial in stereo vision because it provides a measure of the depth of objects in a scene:

- Greater Disparity: Indicates that the object is closer to the cameras. As an object approaches the camera, its images in the left and right views move farther apart.
- Lesser Disparity: Indicates that the object is farther away. As an object moves further from the camera, its projection on the left and right images becomes more aligned.

The baseline b affects the sensitivity of disparity to depth changes.
A larger baseline can increase the disparity for the same depth difference, making the system more sensitive to depth changes but also potentially introducing more errors if the disparity measurement is noisy or imprecise.

The assumption that we made that images are aligned so that their imaging planes are parallel reduces our two dimensional problem to a one dimensional along the same axis.
In fact in this context disparity is just a measure of horizontal shift between corresponding poins in the left and right stereo images.

Template matching can be used to find the disparity by matching a small window or patch around a point in one image (usually the left) to a corresponding area along the same horizontal line in the other image (right).

To perform a disparity calculation:

1. Select a Window in the Left Image: Choose a small region or window around a point of interest in the left image. This window should be large enough to contain distinctive features for matching but not so large that it includes multiple unrelated features.
2. Define the Search Range: Set the range of horizontal shifts (disparity range) to search in the right image. This range might be limited by the expected maximum and minimum distances of objects from the cameras.
3. Slide the Template Across the Right Image: For each possible disparity within the defined range, shift the window from the left image across the corresponding horizontal line in the right image.
Calculate Similarity for Each Position: Use a similarity metric to evaluate how well the window from the left image matches each shifted position in the right image. Common metrics include:
    $$
    SAD(u, v, d) = \sum_{x,y} |I_L(x, y) - I_R(x - d, y)| \\
    SSD(u, v, d) = \sum_{x,y} |I_L(x, y) - I_R(x - d, y)|^2 \\
    NCC(u, v, d) = \frac{\sum_{x,y} (I_L(x, y) \times I_R(x - d, y))}{\sqrt{\sum_{x,y} I_L^2(x, y) \times \sum_{x,y} I_R^2(x - d, y)}}
    $$
4. Select the Best Match: The position that results in the highest similarity (lowest difference or highest correlation) is chosen as the match, and the disparity at this position is recorded.
5. Repeat for Other Points: Repeat this process for other points or windows across the image to build a disparity map of the entire scene.

The size of the window used in stereo matching should be chosen based on several factors:

- Scene Content: More textured and detailed areas might require smaller windows to capture fine details, while uniform areas might benefit from larger windows to gather enough data for reliable matching.
- Disparity Gradient: In areas where the disparity changes rapidly (near edges or boundaries of objects), smaller windows can provide more precise disparity estimates.
- Noise Level: Larger windows can help average out noise, but they might also include multiple depth planes, leading to errors in areas with depth discontinuities.

Typically however the size of the size of the window is choosen to be variable.

Certain types of photos and scenarios can create challenges in accurately determining disparity:

- Low Texture Areas: Photos with large, uniform areas (like clear skies or blank walls) provide very little information for disparity calculation because there are few distinctive features to match between images.
- Repetitive Patterns: Images that contain repetitive patterns (like grids, stripes, or similar textures) can lead to ambiguity in matching because multiple areas look similar, making it difficult to identify the correct correspondences.
TODO Stereo matching, improved stereo matching

### Uncalibrated stereo

Uncalibrated stereo vision does not require prior knowledge of the cameras' parameters.
Instead, it estimates the necessary parameters directly from the images using computer vision algorithms.
This approach is suited for dynamic environments where calibration is impractical.

#### Epipolar geometry

Epipolar geometry is a fundamental concept in stereo vision and multiple-view geometry that describes the geometric relationship between two views captured by cameras observing a 3D scene.
It's essential because it simplifies the problem of finding corresponding points between images, which is crucial for tasks like 3D reconstruction, motion estimation, and object recognition.

The fundamental matrix is a 3x3 matrix that encapsulates the epipolar geometry between two images.
Given a point in one image, the corresponding epipolar line in the other image can be computed as a product of the fundamental matrix with the point.
This matrix is central in uncalibrated stereo vision because it can be computed directly from image correspondences without knowing the cameras' intrinsics or extrinsics.

For calibrated systems where the camera intrinsics are known, the essential matrix, derived from the fundamental matrix, directly encodes the rotation and translation (extrinsics) between two camera views.
It provides a more constrained, robust basis for estimating the cameras' relative pose.

The key elements of epipolar geometry are:

- Epipolar Plane: Any plane that contains the baseline (the line connecting the centers of the two cameras) and a point in the scene observed by both cameras.
- Epipole: The point of intersection of the line joining the camera centers (baseline) with the image plane in each camera. Each camera's epipole is where the other camera's center projects onto that camera's image plane.
- Epipolar Line: The intersection of the epipolar plane with the image plane. Each point in one image has its corresponding epipolar line in the other image. These lines constrain the search for corresponding points to one dimension (along the line), rather than two, significantly simplifying the matching process.

##### Essential Matrix

The epipolar constraint is a fundamental concept in stereo vision that relates corresponding points in stereo image pairs through the essential matrix.
This constraint encapsulates the geometric relationship dictated by the cameras' configuration and the 3D structure of the scene.
The epipolar constraint specifies that the corresponding points in two different views must lie on corresponding epipolar lines.
This is derived from the physical configuration of the cameras and the scene geometry. If you know the projection of a point in one image, the epipolar constraint restricts its possible location in the second image to a line, which is the intersection of the epipolar plane (defined by the point and the two camera centers) with the image plane.
Given two corresponding points x in the first image and x' in the second image, and the essential matrix, the epipolar constraint can be expressed as:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$

Consider a 3D point X in the world space.
The first camera captures it as x and after some rotation and  after undergoing a rigid transformation the second camera captures it as x'
This rigid body transformation can be mathematically expressed as

$$
\mathbf{x}' = R(\mathbf{x} - \mathbf{t})
$$

The coplanarity condition dictates that the point X, and the centers of the two cameras (let's denote them as C and C′), must lie in a single plane.
The coplanarity condition can be expressed using the cross product in vector algebra.
For a point X in 3D space that is observed as x in the first camera and x′ in the second camera, the vectors corresponding to these projections, along with the translation vector t between the two cameras, must satisfy the following condition:

$$
(\mathbf{x} - \mathbf{t})^T (\mathbf{t} \times \mathbf{x}) = 0
$$

The cross product $t \times x$ can be expressed as the skew-symmetric matrix

$$
[\mathbf{t}]_{\times} = \begin{bmatrix} 0 & -t_3 & t_2 \\ t_3 & 0 & -t_1 \\ -t_2 & t_1 & 0 \end{bmatrix}
$$

Substituting this into the coplanarity condition, we get:

$$
(\mathbf{x}'^T R) ([\mathbf{t}]_{\times} \mathbf{x}) = 0
$$

This simplifies to:

$$
\mathbf{x}'^T (R [\mathbf{t}]_{\times}) \mathbf{x} = 0
$$

The expression $R [\mathbf{t}]_{\times}$ defines the essential matrix E:

$$
E = R [\mathbf{t}]_{\times}
$$

Therefore we could write the epipolar constraint in terms of the essential matrix as:

$$
\mathbf{x}'^T E \mathbf{x} = 0
$$

Here are some properties of the essential matrix E:

1. Longuett-Higgins equation: $\mathbf{x}'^T E \mathbf{x} = 0$ This property states that any pair of corresponding points x in the first image and x′ in the second image must satisfy this fundamental epipolar constraint.
2. Epipolar lines: For a point $x$ in the first image and a point x' in the second image:
    $$
    l' = E\mathbf{x} \quad l = E^T\mathbf{x}'
    $$
    These equations define the epipolar lines in each image for a point in the other image.
3. Epipoles: For the epipole e' in the first image and the epipole e in the second image
    $$
    e'^T E = 0 \quad
    Ee = 0
    $$
    The epipoles are null spaces for E and E^T.

The essential matrix has 5 degrees of freedom because rotation has 3 and translation has 2.
However the essential matrix has rank 2, because the ske-symmetric matrix has rank 2 and when combined with rotation it retains this rank.
The structure of E ensures that the two non-zero singular values are equal, reflecting the preserved geometric relationships dictated by the rotation and the skew-symmetric nature of [t]_x
​

##### Foundamental matrix

The Fundamental Matrix F is an extension of the Essential Matrix E used in uncalibrated stereo vision setups where the intrinsic camera parameters (i.e., the camera matrix K) are not necessarily known or set to the identity matrix.

The Fundamental Matrix F encapsulates the intrinsic and extrinsic parameters of a stereo camera setup.
It is a 3×3 matrix that satisfies the following epipolar constraint for any pair of corresponding points x and x′ in the two images:

$$
\mathbf{x}'^T F \mathbf{x} = 0
$$

This equation means that the point x′ in one image must lie on the epipolar line corresponding to the point x in the other image as dictated by F.
When the cameras intrinsics K and Ki are known, the foundamental matrix can be derived from the essential matrix E as follows:

$$
F = K'^{-T} E K^{-1}
$$

This setup implies that F depends on both the geometry of the camera setup (information contained in E) and the cameras internal characteristics,  given by K and K'.

Here are some properties of the essential matrix E:

1. Longuett-Higgins equation: $\mathbf{x}'^T F \mathbf{x} = 0$ This property states that any pair of corresponding points x in the first image and x′ in the second image must satisfy this fundamental epipolar constraint.
2. Epipolar lines: For a point $x$ in the first image and a point x' in the second image:
    $$
    l' = F\mathbf{x} \quad l = F^T\mathbf{x}'
    $$
    These equations define the epipolar lines in each image for a point in the other image.
3. Epipoles: For the epipole e' in the first image and the epipole e in the second image
    $$
    e'^T F = 0 \quad
    Fe = 0
    $$
    The epipoles are null spaces for F and F^T.

F has 8 degrees of freedom.
This is because F is a 3×3 projective transformation matrix, which would typically have 9 elements.
However, since F is defined up to a scale (i.e., multiplying all elements by a non-zero scalar gives an equivalent matrix), it effectively has 8 independent parameters.
This is common in projective geometry, where transformations are not absolute but rather defined relative to scale.
Both E and F are specified to have rank 2.
The rank of E is 2 because it is formed by the product of a rotation matrix R and a skew-symmetric matrix [t]_x.
Skew-symmetric matrices derived from cross products have zero as one of their eigenvalues, which implies that they have rank 2 (only two non-zero singular values).

To compute the foundamental matrix from a set of point correspondences between the two stereo images the eight points algorithm is used.
It requires at least eight point correspondences between two images. The algorithm exploits the linear nature of the relationship described by the epipolar constraint.
The algorithm goes as follows:

0. To improve numerical stability, first normalize the point coordinates. This involves translating and scaling the points so that the centroid of the points is at the origin and the average distance to the origin is sqrt2
1. Describe the problem as a system of equations that satisfy the epipolar constraint for F.
2. Each point correspondence provides one linear equation in the entries of F. Each point correspondence can be written as:
    $$
    x'_m x_m f_1 + x'_m y_m f_2 + x'_m f_3 +\\
    y'_m x_m f_4 + y'_m y_m f_5 + y'_m f_6 + \\
    x_m f_7 + y_m f_8 + f_9 = 0
    $$
    Which can be written with a compact matrix representation:
    $$
    [x'_m x_m, x'_m y_m, x'_m, y'_m x_m, y'_m y_m, y'_m, x_m, y_m, 1] \begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \\ f_5 \\ f_6 \\ f_7 \\ f_8 \\ f_9 \end{bmatrix} = 0
    $$
3. We can ensenble the matrix A, with M >= 8 point corrispondences we can asemble the matrix A:
    $$
    A = \begin{bmatrix}
    x'_1 x_1 & x'_1 y_1 & x'_1 & y'_1 x_1 & y'_1 y_1 & y'_1 & x_1 & y_1 & 1 \\
    x'_2 x_2 & x'_2 y_2 & x'_2 & y'_2 x_2 & y'_2 y_2 & y'_2 & x_2 & y_2 & 1 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    x'_M x_M & x'_M y_M & x'_M & y'_M x_M & y'_M y_M & y'_M & x_M & y_M & 1
    \end{bmatrix}
    $$
4. Solve the system using SVD. First compute the SVD of A
    $$
    A = U \Sigma V ^T
    $$
    in which the solution is the smallest singular value.
5. Enforce the Rank-2 constraint on F:
    $$
    F = U_F \Sigma_F V_F^T \quad \text{where} \quad \Sigma_F = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & 0 \end{bmatrix}
    $$
6. unnormalize F

###### Triangulation

Triangulation is a crucial concept in stereo vision used to determine the three-dimensional coordinates of a point from its projections in two (or more) images taken from different viewpoints.
Triangulation involves identifying the 3D point that corresponds to a specific pair of 2D points observed from two different camera positions.

Given a set of (noisy) matched points {x_i, x_i'} and camera matrices {P,P'} our goal is to esimate the 3d point X.

We start by remembering that since the projection matrixes are known the projection equation is given by:

$$
\mathbf{x} = \mathbf{P} \mathbf{X}
$$

In real-world scenarios, we don't observe the homogeneous coordinates directly in the image. We only observe the scaled 2D coordinates x.
So, the relationship between the projected 2D point and the 3D point can be written with a scale factor \alpha:

$$
\mathbf{x} = \alpha \mathbf{P} \mathbf{X}
$$

Here \alpha is the unknown scale factor that relates the 2D coordinates to the homogenous projection, this can be interpreted as x and PX  are collinear (they point in the same direction but differ by a scale factor).

To get rid of the scale factor we use a geometrical trick: the cross product of two collinear vectors is zero:

$$
\mathbf{x} \times (\mathbf{P} \mathbf{X}) = 0
$$

Assuming that x = [x y 1]^T the cross product of x and PX can be expanded as:

$$
\mathbf{x} \times \mathbf{P} \mathbf{X} = \begin{bmatrix}
y p_3^T \mathbf{X} - p_2^T \mathbf{X} \\
p_1^T \mathbf{X} - x p_3^T \mathbf{X} \\
x p_2^T \mathbf{X} - y p_1^T \mathbf{X}
\end{bmatrix} = \mathbf{0}
$$

Where p_1^T, p_2^T and p_3^T are the rows of the projection matrix P.
This gives us three euqations for each point x, but because of homogeneity, only two are indipendent.

Now, for each 2D-3D point correspondence, we get two independent equations.
For a single image point x and its corresponding 3d point X, the system of equations looks like this:

$$
\begin{bmatrix}
y p_3^T - p_2^T \\
p_1^T - x p_3^T
\end{bmatrix} \mathbf{X} = \mathbf{0}
$$

If you have multiple cameras (or two views), you can concatenate the equations from each view to form a larger system.
For example, in our case, we have two cameras with  two projection matrices:

$$
\begin{bmatrix}
y_1 p_{13}^T - p_{12}^T \\
p_{11}^T - x_1 p_{13}^T \\
y_2 p_{23}^T - p_{22}^T \\
p_{21}^T - x_2 p_{23}^T
\end{bmatrix} \mathbf{X} = \mathbf{0}
$$

This is a homogenoeus system in the form AX = 0.
To find the 3D point X we solve the system.
Since it is a homogeneous system, the solution is typically found using Singular Value Decomposition (SVD).
The solution to such a system is the smallest singular value of A.

In an ideal scenario without noise, the rays projected from different cameras through the corresponding points (obtained via feature matching) should intersect exactly at the location of the 3D point.
However, due to noise in the data, these rays often do not intersect perfectly.
This non-intersection is typically caused by small errors in determining the exact position of the points in each image (due to pixel quantization, camera calibration errors, etc.).

Since the rays do not intersect perfectly due to noise, the least squares method is used to find the best possible point that minimizes the error in terms of the distance from all rays.
This method computes an optimal solution that is the closest to satisfying all the equations given by the projection matrices.
Mathematically, it involves setting up a system of equations derived from each camera's projection equation and solving them by minimizing the sum of the squared differences (errors) between the observed projections and the projections predicted by the model.
In practice, to better handle noise and refine the results, a non-linear least squares method known as bundle adjustment is often employed.

##### Image rectification

The main goal of rectification is to transform the image planes of two cameras so that the epipolar lines become aligned horizontally.
This alignment ensures that the corresponding points in the two images lie on the same horizontal line, greatly simplifying the disparity computation needed for depth estimation.

Each camera captures a different perspective of the same 3D scene.
The epipolar geometry between these two views describes the intrinsic projective geometry between them, characterized by epipolar lines and epipoles (where the epipolar lines converge).
To modify the images such that all epipolar lines are parallel to the horizontal axis of the images.
his is done by applying a projective transformation (homography) to each image.

The fundamental or essential matrix is decomposed to extract the rotation (R) and translation (t) vectors between the two camera views.
The essential matrix E can be decomposed using Singular Value Decomposition (SVD)
$$
\mathbf{E} = \mathbf{U} \Sigma \mathbf{V}^T
$$
Using the rotation matrices obtained from the decomposition, rectifying homographies are calculated for each camera. These homographies are designed to reproject the images so that the epipolar lines become parallel to the horizontal axis.
The original images are warped using the calculated homographies, resulting in rectified images where the stereo correspondence problem is reduced to a one-dimensional search along horizontal lines.

The rotation matrix for rectification, R_rect, is constructed to align the epipoles along the horizontal axis. This involves setting the new x-axis (using vector r1) in the direction of the baseline (the line connecting the camera centers), the y-axis (vector r_2) orthogonal to r_1 in the horizontal plane, and the z-axis (vector r_3) as the cross-product of r_1 and r_2, ensuring a right-handed coordinate system.

$$
r_1 = \frac{\mathbf{t}}{\|\mathbf{t}\|} \\
r_2 = \frac{(0, 0, 1) \times r_1}{\|(0, 0, 1) \times r_1\|}
r_3 = r_1 \times r_2
$$

By ensuring the cameras’ image planes are coplanar and parallel to each other in their rectified state, matching points between the two images only requires comparing points along the same scanlines.

## Template matching

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

## Stereo block matching

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

## disparity search algorithm

DSI encapsulates the calculated match scores between patches of pixels from two rectified stereo images (typically left and right images).
Each pixel on a scanline in the left image is compared against pixels on the corresponding scanline in the right image over a range of disparities.
Think of the DSI as a matrix where one dimension represents the pixels along a scanline of the left image, and the other dimension represents possible disparities.
Each element in this matrix, C(i,j), corresponds to a match score between a patch centered around pixel i in the left image and a patch centered at pixel j in the right image where j varies from i−d to i+d within a feasible disparity range.

The primary use of the DSI is to facilitate the computation of the optimal disparity for each pixel, which in turn helps in generating a depth map of the scene. Depth maps are critical in applications such as 3D reconstruction, autonomous navigation, and virtual reality.
A simple way to use the DSI is through a greedy algorithm where, for each pixel i in the left image, the disparity j that minimizes the dissimilarity (or maximizes the similarity) is chosen. This corresponds to finding the minimum in each column of the DSI matrix if searching along one dimension.
Visually, the DSI can be displayed as an image where each column represents the disparity scores for a specific pixel in the left image across all tested disparities.
Darker values might represent lower costs (better matches), and brighter values indicate higher costs (poorer matches).

## Stereo matching

### local

Local stereo matching algorithms calculate disparity based on the similarity of local image patches. These methods typically follow these steps:

Disparity Range: Set a range of possible disparities.
Block Matching: For each pixel, compare a block (or patch) of pixels around it in one image to corresponding blocks in the other image over the range of disparities.
Cost Calculation: Calculate a similarity score (such as Sum of Absolute Differences, SAD) for each block comparison.
Disparity Selection: Assign to each pixel the disparity that results in the lowest cost, often using a winner-takes-all strategy.
This approach is relatively simple and fast but often falls short in areas with low texture or repetitive patterns due to relying solely on local information.

### non local

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

### Deep models for stereo matching

#### Siamese networks

Siamese Networks for Stereo Matching are designed to leverage convolutional neural networks (CNNs) to perform patch-wise training on images that have associated ground truth disparity maps, such as those provided by the KITTI Stereo 2015 dataset. The process begins by training the network to calculate features from each image in the stereo pair using a learned model. These features, derived from convolutional layers, capture various abstract representations of the image data.

Once features are extracted, they are correlated between the two images, typically through operations like the dot product. This correlation aims to quantify the similarity between corresponding features across the two images, providing a basis for determining how one image relates to the other spatially.

The final step involves the disparity calculation, where the results of the feature correlation are analyzed to determine the most likely disparity for each pixel. This can be done using a simple winner-takes-all approach where the maximum correlation score dictates the disparity or through more sophisticated global optimization techniques that consider additional constraints and regularization to refine the disparity estimation.

This approach is fundamentally different from traditional methods as it uses the power of deep learning to directly learn from data, making it robust against common issues in stereo vision like occlusions and textureless regions.

#### DispNet

DispNet represents an advanced approach in deep learning for stereo matching, marking its significance as the first end-to-end trained deep neural network specifically designed for this purpose. The architecture of DispNet is inspired by the U-Net model, which features a contracting path to capture context and an expanding path that enables precise localization. This design helps in retaining important details through skip-connections that directly link layers of equal resolution in the contracting and expanding paths.

At the heart of DispNet is a correlation layer that processes disparities up to a certain pixel range (e.g., 40 pixels) to establish a similarity score between patches of the left and right images. This layer effectively captures the potential shifts between corresponding features in the two images, facilitating accurate disparity estimation.

To enhance the performance and ensure robustness across different scenarios, DispNet employs a multi-scale loss function. This approach helps the network learn to predict disparities accurately by gradually introducing it to increasingly challenging scenarios, a technique known as curriculum learning. The network is trained on a large dataset, allowing it to generalize well across various stereo vision challenges.

#### Stereo Mixture Density Networks

Stereo Mixture Density Networks (SMD-Nets) offer a novel approach in the field of stereo matching by leveraging the capabilities of mixture density networks to provide high-resolution disparity predictions. These networks are specifically designed to refine the boundaries between objects in stereo images, resulting in sharper and more detailed disparity maps.

The strength of SMD-Nets lies in their ability to model the probability distribution of possible disparities for each pixel, rather than just predicting a single disparity value. This probabilistic approach allows the networks to better handle ambiguities and occlusions in stereo images, as they can represent multiple potential disparities at each pixel location.

By focusing on higher resolution outputs, SMD-Nets are particularly effective in preserving edge details and the structural integrity of the scene, which are often lost in lower resolution approaches. The application of these networks in stereo vision tasks demonstrates significant improvements in the accuracy and quality of the disparity maps, especially in complex scenes where precise edge details are crucial.
