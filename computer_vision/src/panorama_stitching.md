# Panorama Stitching

Panorama stitching is a crucial technique in computer vision, applied across many fields to solve practical and complex problems. Below are key motivations for its importance:

- **Wider Field of View**: Panorama stitching enables the creation of images with a wider field of view than what can be captured by a single image. This is especially important in fields like real estate, where a comprehensive view of a property is needed, and in robotics, where a broader view assists in navigation and environmental understanding.
- **Higher Resolution**: By combining multiple images, panoramas achieve higher resolution and detail than individual images. This is vital for applications like digital mapping (e.g., Google Street View), where detailed, high-quality imagery is necessary for navigation and information extraction.
- **Immersive Experiences**: Panorama stitching is essential in creating immersive environments for Virtual Reality (VR) and Augmented Reality (AR). These technologies rely on seamless panoramic images to deliver convincing experiences without visual discontinuities.

Panorama stitching enhances the capabilities of imaging devices by enabling broader and more detailed analysis, improving visual content creation, and addressing practical challenges in various industries that rely on computer vision.

## Key Steps in Panorama Stitching

1. **Feature Detection**:  
   The first step in image stitching involves detecting distinctive features in each image. These features should be invariant to changes in scale, rotation, and illumination. Common algorithms include:

   - **SIFT** (Scale-Invariant Feature Transform)
   - **ORB** (Oriented FAST and Rotated BRIEF)

   These algorithms identify keypoints such as corners, edges, or unique patterns that signify significant content changes.

2. **Feature Matching**:  
   After detecting keypoints, the next step is to find corresponding features between different images. This typically involves using feature descriptors—unique signatures that describe features invariant to transformations. Popular algorithms like SIFT and SURF compute descriptors for each keypoint, capturing the local appearance around each point and maintaining invariance to changes in scale, rotation, and lighting.

3. **Transform Model Estimation**:  
   Once features are matched between images, the geometric transformation that aligns one image with another is estimated. Depending on the camera motion, this transformation could be:

   - Simple translation
   - Rotation
   - Complex models like affine or homography transformations

   Algorithms such as **RANSAC** (Random Sample Consensus) are commonly used to estimate the best transformation robustly by iteratively selecting a subset of matches, estimating the transformation, and verifying its alignment across all matches.

4. **Image Warping and Transformation**:  
   After determining the transformation model, it is applied to warp the images to align them. This involves adjusting the pixels of one or more images to ensure that corresponding features in the images match.

5. **Image Blending**:  
   Finally, the aligned images must be blended together to create a seamless panorama. Blending techniques manage overlaps, ensure color consistency, and smooth transitions between images. Common methods include:

   - **Multi-band blending**: Blends images at different scales for better visual quality.
   - **Alpha blending**: Facilitates gradual transitions between images to avoid abrupt changes.

By following these steps, a high-quality panoramic image is produced, enhancing the final result in terms of both visual quality and practical application.
multi-band blending, which blends images at different scales, or alpha blending, which facilitates gradual transitions between images, can be used to enhance the visual quality of the panorama.

## Features in Computer Vision

In computer vision, "features" refer to specific patterns or unique attributes within an image that are important for analyzing and understanding its content. These features vary depending on the task but generally include elements that help differentiate one part of an image from another.

### Common Types of Features

- **Edges**:  
  Edges represent boundaries where sharp changes in brightness occur. They are crucial for detecting objects, boundaries, and shapes within an image.

- **Corners and Interest Points**:  
  Corners are points where two or more edges meet. They are often used because they are invariant to translation, rotation, and changes in illumination. Interest points are distinct, recognizable pixels within the image that can be matched across different images of the same scene or object.

- **Blobs and Regions**:  
  Blobs are regions that differ in brightness, color, or other properties compared to surrounding areas. These are useful for detecting objects that are cohesive in terms of color or intensity.

- **Textures**:  
  Texture refers to the surface variations in intensity that suggest properties like material or surface quality. Texture features are valuable in classifying materials or objects based on their surface characteristics.

### Characteristics of a Good Feature

A good feature in computer vision effectively contributes to tasks such as classification, matching, tracking, or reconstruction. Key characteristics of a good feature include:

- **Distinctiveness**:  
  A good feature should provide enough information to distinguish between different objects or classes, while being robust to irrelevant variations.

- **Invariance**:  
  Features need to be invariant to certain transformations, depending on the application. Common invariances include:
  - **Scale**: The feature should be detectable in both small and large sizes.
  - **Rotation**: The feature should be recognizable regardless of its orientation.
  - **Illumination**: Changes in lighting should not affect feature detectability.
  - **Viewpoint**: The feature should ideally be recognizable from different angles, especially in 3D applications.

- **Repeatability**:  
  The feature should be detectable under varying conditions. If identified in one image, it should be recognizable in another image where the scene appears under different conditions.

- **Efficiency**:  
  For real-time applications, feature extraction and matching must be computationally efficient, ensuring that the process is not prohibitively slow.

### Local vs. Global Features

Features extracted from images can be broadly categorized into:

- **Local Features**:  
  Focus on specific points or regions within the image, often used for tasks like matching or object detection.
  
- **Global Features**:  
  Capture broader, overall characteristics of the image and are typically used for tasks such as scene recognition or image classification.

### Feature Detectors

Feature detectors are algorithms designed to identify points of interest in an image. These points, or "features," are typically locations where the image content changes significantly, such as edges, corners, or blobs. The primary goal of a feature detector is to locate salient points that are distinctive and invariant to transformations like translation, rotation, or scaling.

#### Global Feature Detectors

Global features describe characteristics of the entire image, such as its shape, texture, or color histogram. They summarize the image content as a whole rather than focusing on specific areas, making them ideal for:

- **Image classification**: The whole image defines a category, such as distinguishing between different types of landscapes.
- **Scene recognition**: The overall context or setting of the image is more important than individual elements within it.

#### Local Feature Detectors

Local features capture information about specific points or small regions in the image. These features are particularly useful for tasks that require precise structure or content information from parts of the image. Applications include:

- **Object recognition**: Useful for recognizing objects that appear at different sizes, rotations, or with partial occlusion.
- **Image matching**: Align or stitch images by identifying similar patterns or keypoints across images.

---

##### Harris Corner Detector

The **Harris Corner Detector** is a well-known method for detecting corners, regions where there is a significant change in intensity in multiple directions. Corners are points where two or more edges meet, and detecting them is essential for tasks such as image matching and object recognition.

The algorithm is based on the idea that corners can be detected by analyzing how the image brightness changes when shifted slightly in different directions. This can be quantified using the **summed square difference (SSD)** function, which compares image patches before and after a shift. The intensity change is large around corners but small around edges or flat regions.

The SSD function is defined as:

$$
E_{\text{SSD}}(u) = \sum_{i} [I_1(x_i + u) - I_0(x_i)]^2
$$

Where:

- \( u \) is the small shift vector.
- \( I_1(x_i + u) \) is the intensity at location \( x_i + u \).
- \( I_0(x_i) \) is the original intensity at location \( x_i \).

We can enhance this with a spatial weighting function:

$$
E_{\text{wSSD}}(u) = \sum_i w(x_i) [I_1(x_i + u) - I_0(x_i)]^2
$$

Here, \( w(x_i) \) is a Gaussian windowing function that emphasizes central pixels in the image patch.

###### Small Motion Assumption

Under the small motion assumption, the intensity change due to a small shift \( \Delta u \) is approximated linearly using a Taylor expansion:

$$
I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u
$$

Substituting into the SSD gives us:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(\nabla I(x_i) \cdot \Delta u)^2]
$$

This can be further expanded:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(I_x(x_i) u_x + I_y(x_i) u_y)^2]
$$

This leads to the quadratic form:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) \begin{bmatrix} u_x & u_y \end{bmatrix} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix}
$$

Defining the matrix \( \mathbf{A} \) as:

$$
\mathbf{A} = \sum_{i} w(x_i) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

Matrix \( A \), also known as the **structure tensor**, captures the local image gradients, which help in determining whether a region is a corner, edge, or flat area. The eigenvalues of \( A \) describe the local image structure:

- Large, comparable eigenvalues indicate a corner.
- One large eigenvalue suggests an edge.
- Small eigenvalues correspond to flat regions.

##### Harris Corner Detector Algorithm

1. Compute image gradients \( I_x \) and \( I_y \) (often using a Sobel filter), and smooth them with a Gaussian filter:
    $$
    S_{xx} = G * I_x^2, \quad S_{yy} = G * I_y^2, \quad S_{xy} = G * I_x I_y
    $$
2. Construct the second moment matrix \( A \) using the smoothed gradients:
    $$
    \mathbf{A} = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}
    $$
3. Compute the **corner response function**:
    $$
    R = \det(\mathbf{A}) - k \cdot (\text{trace}(\mathbf{A}))^2
    $$
    Where:
    $$
    \det(\mathbf{A}) = S_{xx} S_{yy} - S_{xy}^2, \quad \text{trace}(\mathbf{A}) = S_{xx} + S_{yy}
    $$

4. Apply a threshold to \( R \) to detect potential corners and use **non-maximum suppression** to keep only local maxima.

##### Properties of the Harris Corner Detector

- **Locality**: Sensitive to local features.
- **Illumination Invariance**: Robust to changes in lighting.
- **Rotation Invariance**: Corners are detected irrespective of image orientation.
- **Partial Scale Invariance**: Can be extended to detect features across scales using a multi-scale approach.
- **Sensitivity to Perspective**: Less robust to perspective transformations without modifications.

---

##### Harris Corner Detector with Scale-Invariance

To make the Harris Corner Detector **scale-invariant**, it can be combined with the **Laplacian of Gaussian (LoG)** filter. This allows detection of features at multiple scales:

1. Apply a Gaussian filter with varying \( \sigma \) values.
2. Compute image gradients and the second moment matrix at each scale.
3. Compute the Harris response at each scale.
4. Perform non-maximum suppression across both spatial and scale dimensions to ensure that detected corners are local maxima.

This approach ensures corners are detected robustly across different scales, adapting to the intrinsic size of the features.

---

##### Blob Detector

A **blob** refers to a region in the image that differs in properties, such as brightness or color, from surrounding areas. The **Laplacian of Gaussian (LoG)** is often used as a blob detector:

1. Apply the Gaussian filter to smooth the image.
2. Use the Laplacian filter to detect regions of rapid intensity change (blobs).
3. Identify **zero crossings** in the Laplacian response, which indicate blob boundaries.

For **scale-invariance**, this process can be applied using a **Gaussian pyramid**:

- At each level of the pyramid, apply the LoG filter to detect blobs at different scales.
- Significant blobs are those that appear consistently across scales.

This method ensures that blob-like features are detected at appropriate scales, making it a powerful tool for detecting regions of interest in an image.

### Feature Descriptors

Feature descriptors are numerical representations of image characteristics, typically extracted from specific regions or points of interest. While feature detectors identify significant points such as corners, edges, or blobs, feature descriptors provide a compact, robust, and informative description of these points. Descriptors aim to be invariant to variations in illumination, rotation, and scale, making them crucial for comparing and matching features across different images.

Once features have been detected, descriptors encapsulate information about the appearance and shape of the surrounding area to allow robust matching across images.

---

#### Building Histogram of Oriented Gradients (HoG)

HoG (Histogram of Oriented Gradients) is a feature descriptor that describes the shape and appearance of objects by analyzing the distribution of gradient orientations. Below is the step-by-step construction of HoG, starting from simple approaches and progressively adding enhancements.

##### Image Patches

- **Basic Approach**: Use raw pixel intensity values inside an image patch.
- **Limitation**: Sensitive to absolute intensity and fails under changes in illumination or geometric transformations like rotation or scaling.

##### Image Gradients

- **Improvement**: Use image gradients (changes in intensity) instead of raw intensity. Gradients are more robust against illumination variations.
- **Limitation**: Gradients improve robustness but do not handle geometric transformations like rotation and scale.

##### Color Histograms

- **Improvement**: Color histograms summarize the distribution of colors, providing robustness against scaling and rotation.
- **Limitation**: Color histograms do not capture spatial information, which can lead to poor matching when spatial arrangement is crucial.

##### Spatial Histograms

- **Improvement**: Divide the image into cells and compute histograms for each cell, capturing spatial layout. This retains both appearance and spatial information, often based on gradients rather than color.
- **Limitation**: Still not fully invariant to rotation.

##### Orientation Normalization

- **Improvement**: Align histograms according to a dominant orientation to achieve rotation invariance.
- **Limitation**: Orientation normalization does not address challenges like object identity based solely on color distribution.

##### HoG Algorithm

HoG combines gradient, spatial structuring, and orientation normalization into a coherent descriptor. The steps to compute HoG are:

1. **Preprocessing**: Convert the image to grayscale to reduce complexity and focus on structure rather than color.
2. **Gradient Computation**: Compute the horizontal and vertical gradients for each pixel, often using a Sobel filter.
3. **Orientation Binning**: Divide the image into cells (e.g., 8x8 pixels). For each cell, create a histogram of gradient orientations, typically with 9 to 18 bins covering 0 to 180 degrees (unsigned) or 0 to 360 degrees (signed).
4. **Descriptor Blocks**: Group adjacent cells into larger blocks (e.g., 2x2 cells). Normalize histograms within each block to reduce sensitivity to lighting variations.
5. **Concatenation**: Combine the normalized histograms from all blocks into a single feature vector representing the HoG descriptor.
6. **Sliding Window**: For object detection, apply HoG within a sliding window across the image.

**HoG Applications**: HoG is particularly effective in human detection, capturing vertical and horizontal edges typical in human forms.

---

#### Scale-Invariant Feature Transform (SIFT)

SIFT is both a feature detector and descriptor, designed to be invariant to scale and rotation while robust against illumination changes. It is widely used for tasks like object recognition and image matching.

##### SIFT Algorithm

1. **Scale-Space Construction**: Construct a scale space by applying Gaussian filters at different scales. The Difference of Gaussians (DoG) is used to detect keypoints.
2. **Keypoint Localization**: Find local maxima and minima in the DoG. Keypoints are refined using Taylor expansion to achieve sub-pixel accuracy.
3. **Orientation Assignment**: Compute the gradient magnitude and orientation around each keypoint. The orientation histogram is created, and the keypoint is assigned a dominant orientation.
4. **Descriptor Computation**:
   - Calculate gradients in a 16x16 region around the keypoint.
   - Rotate the region according to the keypoint’s orientation.
   - Divide the region into 4x4 cells and compute orientation histograms for each.
   - Concatenate the histograms into a 128-element vector.

**SIFT Properties**:

- **Scale and Rotation Invariance**: SIFT features are invariant to changes in scale and orientation.
- **Robustness to Illumination**: The use of gradients makes SIFT somewhat invariant to lighting changes.
- **Distinctiveness**: The 128-element descriptor captures significant local structure, making it highly distinctive.

---

#### Speeded Up Robust Features (SURF)

SURF is a faster alternative to SIFT, designed for speed and efficiency while maintaining robustness. It uses integral images and approximates Gaussian convolutions with box filters for quick computation.

##### SURF Algorithm

1. **Integral Image**: Compute the integral image for fast calculation of box-type filters, which approximate Gaussian filters.
2. **Hessian Matrix**: Use the Hessian matrix to detect keypoints. The determinant of the Hessian matrix highlights blob-like structures in the image.
3. **Keypoint Localization**: Identify local maxima and minima in the determinant of the Hessian matrix across scales.
4. **Descriptor Construction**:
   - Assign orientation based on Haar wavelet responses.
   - Divide the region around the keypoint into 4x4 subregions.
   - Compute Haar wavelet responses in each subregion to capture gradient information.
   - Normalize the descriptor to achieve robustness against lighting changes.

**SURF Properties**:

- **Speed**: Faster than SIFT due to the use of integral images and box filters.
- **Robustness**: Maintains robustness to scale, rotation, and illumination changes.
- **Efficient Matching**: Laplacian indexing enhances matching efficiency by considering the polarity of keypoints.

---

#### Binary Descriptors

Binary descriptors are lightweight and computationally efficient. Instead of using floating-point vectors like SIFT or SURF, binary descriptors create compact binary strings for faster matching.

##### BRIEF (Binary Robust Independent Elementary Features)

- **Approach**: Generate a binary string by comparing intensities of pixel pairs in an image patch.
- **Limitation**: Not rotation invariant and sensitive to noise.

##### ORB (Oriented FAST and Rotated BRIEF)

- **Approach**: Combines the FAST keypoint detector with BRIEF descriptors, adding orientation information for rotation invariance.
- **Efficiency**: ORB is fast and free to use, making it suitable for real-time applications.

##### FREAK (Fast Retina Keypoint)

- **Approach**: Inspired by the human visual system, FREAK compares pixel intensities across a retinal sampling pattern. The pattern is denser near the center and sparser towards the periphery, mimicking retinal ganglion cells.
- **Advantage**: More robust to rotation and scale changes compared to BRIEF, with a lower computational cost than SIFT or SURF.

---

### Feature Matching

Feature matching is the process of identifying corresponding keypoints between different images. It is a critical step in various computer vision applications, such as stereo vision, object recognition, motion tracking, image stitching, and 3D reconstruction. The goal of feature matching is to establish correspondences between features extracted from different images, even when those images are captured under different conditions like varying scale, orientation, or lighting.

#### Key Steps in Feature Matching

1. **Feature Detection**:  
   Features or keypoints are first detected in each image. These points represent areas of the image with significant texture or structure (e.g., corners or edges), which can be reliably identified and are typically invariant to transformations like scaling, rotation, or changes in illumination.

2. **Feature Description**:  
   Each detected feature is described using a feature descriptor. Descriptors encode the appearance of the feature and its surrounding region in a compact form, making them robust against transformations. The goal is for the same feature to have a similar descriptor even when captured under different conditions.

3. **Feature Matching**:  
   Once features are described, the descriptors are matched across different images. This involves finding pairs of descriptors that are closest in terms of a chosen distance metric. Common metrics include:
   - **Euclidean distance**: Used for real-valued descriptors like SIFT and SURF. It calculates the straight-line distance between two points in a high-dimensional space.
   - **Hamming distance**: Used for binary descriptors like ORB and BRIEF. It counts the number of differing bits between two binary strings, making it computationally efficient.

---

#### Matching Techniques

- **Brute-Force Matching**:  
  The simplest method of feature matching, where each descriptor from one image is compared with every descriptor in the second image. While exhaustive and accurate, brute-force matching can be computationally expensive for large datasets.

- **Nearest Neighbor Matching**:  
  In this method, each descriptor from one image is matched with the descriptor from the second image that has the smallest distance. To improve robustness, techniques such as the ratio test (proposed by Lowe in SIFT) can be applied:
  - **Ratio Test**: Compares the distance of the nearest neighbor to the second nearest neighbor. If the ratio of the two distances is below a threshold, the match is accepted; otherwise, it is discarded to avoid false matches.

- **K-Nearest Neighbors (k-NN)**:  
  Instead of finding only the closest match, this method identifies the k closest matches for each feature. The ratio test can also be applied here to filter out ambiguous matches.

- **FLANN (Fast Library for Approximate Nearest Neighbors)**:  
  This is a more efficient alternative to brute-force matching, especially for large datasets. FLANN uses approximate nearest neighbor search algorithms, which trade off a slight loss in accuracy for a significant speed improvement.

---

#### RANSAC for Robust Matching

Once potential matches have been identified, it is important to refine them to reduce outliers. **RANSAC (Random Sample Consensus)** is commonly used to filter out incorrect matches:

- **RANSAC Algorithm**:  
  A subset of the matched points is randomly selected to compute a transformation (e.g., homography for planar surfaces or fundamental matrix for stereo images). The transformation is then applied to all the matched points, and those that fit the transformation are considered inliers. This process is repeated iteratively to maximize the number of inliers and minimize false matches.

---

#### Applications of Feature Matching

- **Image Stitching**:  
  In panorama creation, feature matching aligns overlapping images by finding corresponding points, allowing the images to be stitched together seamlessly.
  
- **Object Recognition**:  
  Feature matching allows for recognizing objects across different images, even when they are rotated or viewed from different angles.
  
- **3D Reconstruction**:  
  By matching keypoints between multiple views of the same scene, depth information can be estimated, enabling the reconstruction of 3D models from 2D images.

- **Motion Tracking**:  
  In video analysis, feature matching helps track objects or features across consecutive frames, facilitating motion tracking or optical flow estimation.

### Deep Learning Approaches

#### SuperPoint and SuperGlue

**SuperPoint** is a convolutional neural network (CNN) that simultaneously detects interest points and computes descriptors. It is pre-trained on synthetic data and fine-tuned on real-world images using self-supervised learning.

- **Dual Head Architecture**: A shared CNN encoder extracts feature maps for both interest point detection and descriptor computation. One head produces a heatmap for keypoint detection, and the other outputs a dense descriptor map for describing keypoints.
- **Keypoint Detection**: Local maxima in the heatmap identify keypoints, with non-maximum suppression ensuring they are well-distributed.
- **Descriptor Generation**: Descriptors, extracted from the corresponding keypoints in the descriptor map, are designed to be distinctive and robust against image transformations.

**SuperGlue** is a feature matching method that enhances SuperPoint's output using a Graph Neural Network (GNN) to improve matching accuracy, especially in challenging conditions (e.g., large viewpoint changes or occlusions).

- **Graph Construction**: Descriptors from SuperPoint (or another detector) are treated as graph nodes, with edges representing potential matches between two images.
- **Graph Neural Network**: The GNN refines matches by considering both local descriptor similarities and the global geometric consistency of features across the graph.
- **Match Filtering**: SuperGlue selects matches that are both locally and globally consistent, outperforming traditional nearest neighbor methods in complex scenarios.

## Image Transformations

In computer vision, a transformation modifies the geometry of an image, including its position, size, orientation, or perspective. A transformation maps points in one image to corresponding points in another, altering their spatial relationships.

### Homogeneous Coordinates

Homogeneous coordinates extend the representation of points by adding an extra dimension, allowing for more complex transformations, such as perspective changes. They also enable affine transformations, including translations, to be represented as matrix operations. Points at infinity, which are useful in perspective transformations, are represented by setting \( w = 0 \).

### Linear Transformations

Linear transformations include operations like translation, rotation, scaling, and shearing. These transformations are simple to compute and can be represented as matrix multiplications.

These include:

- **Rotation**: Changes the orientation of an image.
- **Scaling**: Alters the size of an image.
- **Shearing**: Skews the image along one axis.

They preserve linear relationships and distances between points and are expressed as matrix multiplications.

### Affine Transformations

Affine transformations include all linear transformations plus translation. They maintain parallelism of lines and can be expressed using homogeneous coordinates in matrix form:

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} =
\begin{bmatrix}
a & b & t_x \\
c & d & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

### Homographies

Homographies (projective transformations) extend affine transformations by allowing changes in perspective. They can warp, tilt, and scale image content and are particularly useful for stitching images with different viewpoints. Homographies are represented by a 3x3 matrix that transforms points in homogeneous coordinates:

$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix}=
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

### Computing Transformations

Computing a transformation involves determining the transformation matrix \( T \) that maps points from image \( A \) to image \( B \), given a set of matched feature points between the two images. The goal is to find the matrix that best fits the transformation using methods like least squares, which minimizes the error from imperfect feature matching.

---

#### Computing Translation

Given matched points \( (x_i, y_i) \) in \( A \) and \( (x'_i, y'_i) \) in \( B \), the translation equations are:

$$
x'_i = x_i + t_x \\
y'_i = y_i + t_y
$$

Rearranging gives:

$$
t_x = x'_i - x_i \\
t_y = y'_i - y_i
$$

To compute \( t_x \) and \( t_y \) across all matches, the least squares approach minimizes the sum of squared differences:

$$
\min_{t_x, t_y} \sum_{i=1}^n \left((x'_i - x_i - t_x)^2 + (y'_i - y_i - t_y)^2\right)
$$

The solution for \( t_x \) and \( t_y \) is:

$$
\hat{t}_x = \frac{1}{n} \sum_{i=1}^n (x'_i - x_i), \quad \hat{t}_y = \frac{1}{n} \sum_{i=1}^n (y'_i - y_i)
$$

This averages the translation estimates from all matches, reducing the impact of noise.

---

#### Computing Affine Transformations

For affine transformations, which include translation, rotation, scaling, and shearing, the relationship between matched points is:

$$
x'_i = a \cdot x_i + b \cdot y_i + t_x \\
y'_i = c \cdot x_i + d \cdot y_i + t_y
$$

This introduces six unknowns \( a, b, c, d, t_x, t_y \). The least squares method minimizes:

$$
\min_{a, b, c, d, t_x, t_y} \sum_{i=1}^n \left((x'_i - (a \cdot x_i + b \cdot y_i + t_x))^2 + (y'_i - (c \cdot x_i + d \cdot y_i + t_y))^2\right)
$$

At least three matched points are required to compute the affine transformation.

---

#### Computing Homographies

Homographies allow for perspective transformations, represented by a 3x3 matrix. The general form is:

$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix} =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

The non-linear relationship between input and output coordinates is linearized by cross-multiplying, leading to:

$$
x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13} \\
y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}
$$

Each matched point provides two equations, requiring at least four point pairs to solve for the eight unknowns (assuming \( h_{33} = 1 \) to resolve scale ambiguity).

To minimize error, least squares finds:

$$
\|Ah - b\|^2
$$

Where \( A \) is constructed from the linearized equations, and \( h \) is the homography vector. Singular Value Decomposition (SVD) is used to solve this overconstrained system, where the optimal \( h \) is the eigenvector of \( A^T A \) corresponding to the smallest eigenvalue, minimizing the residual error.

##### RANSAC

When computing transformations between two images, such as a homography between image \( A \) and image \( B \), feature matches are crucial. However, due to detection errors, scene changes, and mismatches, outliers can occur—pairs of points that do not correspond to the same physical point in space. Outliers can severely distort the computed transformation, leading to inaccurate results when using least squares optimization.

**RANSAC (Random Sample Consensus)** is a robust algorithm used to estimate the transformation while mitigating the effects of outliers. RANSAC iteratively tries to find a model that best fits the inliers, ignoring the outliers. The steps are:

1. **Sample Minimum Points**: Randomly select the minimal number of point pairs required to estimate the transformation (e.g., 4 pairs for homography).
2. **Fit Model**: Compute the transformation (e.g., homography) from the selected points.
3. **Evaluate Inliers**: Apply the model to all matches and count the inliers—points that conform to the model within a defined error threshold.
4. **Iterate**: Repeat the process, each time with different random point pairs, to find the transformation with the highest number of inliers.
5. **Select Optimal Model**: Choose the model with the most inliers.
6. **Refine Model**: Recalculate the transformation using only the inliers to improve accuracy.

The number of RANSAC iterations \( N \) needed to achieve a high probability \( p \) of success is determined by:

$$
N = \frac{\log(1 - p)}{\log(1 - (1 - e)^s)}
$$

Where:

- \( s \) is the number of points needed for the model (e.g., 4 for a homography),
- \( e \) is the proportion of outliers,
- \( p \) is the desired probability of finding a valid model.

---

**Pros of RANSAC**:

- **Robustness**: Effectively handles datasets with a high proportion of outliers.
- **Generality**: Can be applied to various model-fitting tasks beyond homographies, such as fundamental matrices and affine transformations.

**Cons of RANSAC**:

- **Computational Cost**: High iteration count may be required when the proportion of outliers is large, leading to slower computations.
- **Failure in Dense Outliers**: If outliers dominate the dataset, RANSAC may struggle to find a valid model.

## Warping

After computing the homography between two images, warping is the process of applying this transformation to align or stitch the images. Warping modifies the coordinates of pixels to fit a new coordinate system. There are two main types:

- **Forward Warping**: Directly applies the homography to each pixel in the source image, mapping it to a new position in the destination image. This can lead to issues like gaps (where no pixels are mapped) or overlaps (where multiple pixels map to the same destination). Post-processing is required to fill gaps or resolve overlaps, potentially affecting image quality.
  
- **Inverse Warping**: Iterates over each pixel in the destination image and uses the inverse of the homography to map back to the source image. This method avoids gaps by ensuring all destination pixels are accounted for and handles overlaps naturally.

Both methods involve interpolation since transformed coordinates often map to non-integer positions.

---

## Blending

Blending is essential in panorama stitching to smooth the transitions between overlapping images and avoid visible seams, color inconsistencies, or exposure differences. This is especially important when images were taken under varying conditions.

### Common Blending Techniques

- **Feathering**:
  - **Weighted Mask**: A gradient mask is applied across the overlap, gradually blending the images. Pixels near the edge of the overlap have lower weight, while those toward the center have higher weight.
  - **Linear Interpolation**: The pixel values in the overlap are computed by linearly interpolating between corresponding pixel values, weighted by the mask. This reduces visible seams but may cause ghosting if the images aren’t perfectly aligned.

- **Pyramid Blending**:
  - **Image Pyramids**: Each image is decomposed into a multi-scale pyramid (e.g., using Gaussian pyramids) with progressively lower resolutions.
  - **Blend Each Layer**: Corresponding layers from each pyramid are blended separately.
  - **Reconstruct**: The final image is reconstructed from the blended pyramid layers. This method handles variations across different scales, reducing ghosting and improving smooth transitions.

- **Laplacian Pyramid Blending**:
  - **Laplacian Pyramids**: These pyramids capture image details by subtracting each Gaussian pyramid level from the next, isolating finer details at each scale.
  - **Blend and Reconstruct**: The Laplacian pyramids are blended, and the final image is reconstructed from the combined pyramid. This technique excels at preserving edge details, making it ideal for images with complex overlaps.
