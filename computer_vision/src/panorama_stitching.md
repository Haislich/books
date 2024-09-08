# Panorama stritching

Panorama stitching is a significant technique in computer vision for several reasons, and understanding its importance can give insights into how computer vision is applied to solve practical and complex problems.

Here are some motivations on why panorama stitching matters:

- Panorama stitching allows for the creation of images with a much wider field of view than what is possible with a single image capture. This is crucial in fields like real estate, where panoramic images can provide a more comprehensive view of a property, or in robotics, where a wider view can help in navigation and environment understanding.
- By combining multiple images, panoramas can achieve higher resolution and detail than individual images. This is especially useful in applications like digital mapping (e.g., Google Street View) where detailed, high-quality images are necessary for both navigation and information extraction.
- Panorama stitching is fundamental in creating immersive environments for VR and AR applications. These technologies rely on seamless panoramic images to provide users with convincing immersive experiences that are free from visual discontinuities.

In essence, panorama stitching enhances the capabilities of individual imaging devices and techniques. It enables broader and more detailed analysis, improves visual content creation, and addresses practical challenges across various disciplines and industries within computer vision.

1. Feature Detection: The initial step in image stitching involves detecting distinctive features in each image. These features should be invariant to changes in scale and rotation and, ideally, to variations in illumination as well. Algorithms such as SIFT (Scale-Invariant Feature Transform) or ORB (Oriented FAST and Rotated BRIEF) are commonly utilized for this purpose. They identify keypoints that signify significant changes in content, such as corners, edges, or other unique patterns.
2. Feature Matching: After keypoints are detected in each image, the next step is to find corresponding features between different images. This typically involves using feature descriptors, which are unique signatures that describe the features in a manner invariant to various transformations. Algorithms like SIFT and SURF also compute descriptors for each keypoint. These descriptors capture the local appearance around each point and are designed to be invariant to changes in scale, rotation, and lighting.
3. Transform Model Estimation: Once features are matched between images, the next task is to estimate the geometric transformation that aligns one image with another. This transformation could be a simple translation, a rotation, or more complex models like affine or homography transformations, depending on the camera motion. RANSAC (Random Sample Consensus) or similar algorithms are used to robustly estimate the best transformation. This method operates by iteratively selecting a subset of matches, estimating the transformation, and verifying how well the transformation aligns all matches.
4. Image Warping and Transformation: After establishing the transformation model, the next step is to apply this transformation to warp the images so that they align correctly. This step adjusts the pixels of one or more images to ensure that features from one image match the corresponding features in the other image.
5. Image Blending: Finally, once the images are aligned, they must be blended together to create a seamless panorama. This process involves managing overlaps, ensuring color consistency, and smoothing the transitions between images. Techniques like multi-band blending, which blends images at different scales, or alpha blending, which facilitates gradual transitions between images, can be used to enhance the visual quality of the panorama.

## Features

In computer vision, "features" refer to specific patterns or unique attributes within an image that are important for analyzing and understanding the image's content.
These features can vary widely depending on the task but generally include elements that help in differentiating one part of an image from another.

Commonly used features in computer vision are:

- Edges: These are boundaries where there are sharp changes in image brightness. Edges can be used to detect objects, boundaries, and shapes within an image.
- Corners and Interest Points: Corners are points where two or more edges meet, and they are typically used because they are invariant to translation, rotation, and illumination changes. Interest points are distinct and easily recognizable pixels in the image that can be used for matching different images of the same scene or object.
- Blobs and Regions: Blobs are regions of an image that differ in properties, like brightness or color, compared to surrounding regions. These are useful for detecting and recognizing objects when the objects are cohesive in color or intensity.
- Textures: Texture describes the variation in the intensity of a surface that suggests properties about the material or quality of the surface. Texture features can help in classifying materials or objects based on their surface characteristics.

A good feature in computer vision is one that effectively contributes to the specific task at hand, whether it's classification, matching, tracking, or reconstruction. The qualities that make a feature "good" can vary depending on the application, but generally, a good feature has several key characteristics:

- A good feature should clearly differentiate between different classes or objects. It should provide enough information to distinguish between the variations that are relevant to the task while being robust to irrelevant variations.
- Features need to be invariant to certain transformations depending on the application requirements. Common invariances include:
  - Scale: The feature should be detectable in both small and large sizes.
  - Rotation: The feature should be recognizable no matter its orientation.
  - Illumination: Changes in lighting should not affect the detectability of the feature.
  - Viewpoint: Ideally, the feature should be recognizable from different angles, especially in 3D applications.
- A feature should be detectable under different conditions in the same way. If an algorithm identifies a feature in one image, it should be able to identify the same feature in another image where the scene or object appears under slightly different conditions.
- The computation of the feature and its matching or recognition process should be efficient enough for the application, especially in real-time systems. This means that feature extraction and processing should not be prohibitively slow.

In computer vision, features extracted from images can broadly be categorized into local and global features, each serving distinct purposes depending on the application.

### Feature detectors

Feature detectors are algorithms designed to identify points of interest within an image.
These points, or features, are typically locations where the image content changes significantly, such as edges, corners, or blobs.
The primary goal of a feature detector is to locate these salient points that are distinctive and invariant to various transformations such as translation, rotation, or scaling.

#### Global feature Detectors

Global features describe characteristics of the whole image, such as its shape, texture, or color histogram. These features are not specific to a particular area of the image but rather summarize the entire content. Global features are ideal for:

- Image classification where the entire image defines a category (like distinguishing between different types of landscapes).
- Scene recognition where the overall context or setting of the image is more important than individual elements within it.

#### Local Feature Detectors

Local features refer to the properties or patterns found in small parts of an image.
They capture information about the image at specific points or small neighborhoods, making them very useful for tasks where the precise structure or detailed content within a part of an image is significant. These features are particularly useful for:

- Object recognition where the object might appear in different sizes, rotations, or with partial occlusions.
- Image matching where you need to align or stitch images based on similar patterns or specific keypoints.

##### Harris corner detector

The Harris Corner Detector is a popular method used in computer vision for detecting corners, which are regions in an image with significant changes in intensity in multiple directions.
The Harris Corner Detector, introduced by Chris Harris and Mike Stephens in their 1988 paper, is based on the idea that corners can be detected by looking for significant changes in image brightness when the image is shifted by a small amount in any direction.
A corner can be defined as a point in the image at which two or more edges meet, or where there is an intersection of discontinuities in the image brightness.
These discontinuities generally indicate a variation in the image gradient in multiple directions.
Corners are expected to yield a large change in appearance in such shifts, unlike edges (which change in one direction) or flat regions (which change little in any direction).

We can formalize this idea by using the summed square difference function (SSD), which is an error function quantifies the difference between two image patches, typically comparing a patch shifted by a small vector $u$ against its original position.
The goal is to see how 'stable' a window (or image patch) is when shifted slightly in different directions.
The less the intensity values change, the lower the SSD, which usually indicates flat or uniform areas.
In contrast, high SSD values suggest significant intensity changes, as seen around corners.

$$
E_{\text{SSD}}(u) = \sum_{i} [I_1(x_i + u) - I_0(x_i)]^2
$$

This function helps determine how the intensity of a window changes when it's slightly shifted, indicating potential feature points like corners or edges based on the variability of the result.

We can incorporate the information from a spatial weighting function to enhance the SSD by focusing more on the central pixels:

$$
E_{\text{wSSD}}(u) = \sum_i w(x_i) [I_1(x_i + u) - I_0(x_i)]^2
$$

Where $w$ is a Gaussian windowing function.

The small motion assumption posits that the movement (or displacement) between consecutive frames or images is small enough that the changes in the image can be approximated linearly.
This assumption is crucial when analyzing changes in an image over time or across slightly different viewpoints.

Under the small motion assumption, the change in intensity of a pixel due to a small shift can be approximated using a Taylor series expansion, which is truncated after the linear term.
For an image $I$ and a small motion $\delta u$ at a pixel location $x_i$ the intensity at the new location can be approximated as:

$$
I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u
$$

Where $\nabla I(x_i)$ represents the image gradient at $x_i$.

$$
E_{wSSD}(\Delta u) = \sum_{i} w(x_i) [I(x_i + \Delta u) - I(x_i)]^2
$$

Applying the small motion assumption, where $\Delta u$ is small, the change in $I$ due to $\Delta u$ is small and can be aproximated linearly as:

$$
I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u
$$

Which substituting into the error function gives us:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(I(x_i) + \nabla I(x_i) \cdot \Delta u - I(x_i))^2]
$$

Symplifying, gets us:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(\nabla I(x_i) \cdot \Delta u)^2]
$$

Finally expanding the dot product we get:

$$
wSSD(\Delta u) = \sum_{i} w(x_i) [(I_x(x_i) u_x + I_y(x_i) u_y)^2]
$$

This equation, has a form of a quadratic expression in terms of

$$
wSSD(\Delta u) = \sum_{i} w(x_i) \begin{bmatrix} u_x & u_y \end{bmatrix} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix}
$$

Which leads to define the matrix $A$ as:

$$
\mathbf{A} = \sum_{i} w(x_i) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

The matrix $A$ is commonly referred to as the second moment matrix or the structure tensor.
This matrix encapsulates important information about the intensity gradients at a given point in the image, offering insights into the local image structure.
A  is composed of the sums of the squares and cross-products of the image derivatives (gradients) at each point, weighted by a window function (often Gaussian).
This weighting emphasizes contributions from pixels closer to the center of the window, aligning with the intuition that the central region should have more influence in determining whether a point is a corner.

- $I_x^2$ aggregates the squared gradients in the horizontal direction within the window, indicating the extent of horizontal edges.
- $I_y^2$ aggregates the squared gradients in the vertical direction, indicating the extent of vertical edges.
- $I_xI_y$ captures the product of horizontal and vertical gradients, indicating the presence of mixed or diagonal features.

In statistics, a covariance matrix provides a measure of the strength of the correlation between two or more sets of random variates.
The diagonal entries of a covariance matrix represent the variance of each variable, while the off-diagonal entries represent the covariance between the variables.
In the context of image processing, particularly in the Harris Corner Detector, the matrix $A$ can be thought of as a covariance matrix if we consider the image gradients $I_x$ and $I_y$ as random variables across the image window.
Te entries $I_x^2$, $I_y^2$ and $I_xI_y$  are analogous to the variances and covariance found in a statistical covariance matrix, describing how much the gradient in one direction varies with the gradient in another direction within the window.
Viewing $A$ as a covariance matrix provides a statistical insight into how variable the image intensity is in different directions, which is fundamental for understanding the underlying image structure and detecting features.

In Harris Corner Detection, the eigenvalues of $A$ determine whether a pixel is a corner, edge, or flat area.
Large and comparable eigenvalues indicate corners, a single large eigenvalue indicates an edge, and small eigenvalues indicate flat areas.

Here's the complete harris corner detector algorithm:

1. Calculate image gaussian gradients along the orizontal and vertical directions, typically using the sobel filter and appyling afterwards a gaussian kernel. This last step is crucial for adding locality to the response, emphasizing the contributions from pixels closer to the central pixel.
    $$
    S_{xx} = G * I_x^2, \quad S_{yy} = G * I_y^2, \quad S_{xy} = G * I_x I_y
    $$
2. Assemble the second moment matrix $A$ using the smoothed gradients.
    $$
    \mathbf{A} = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}
    $$
3. Determine the corner response at each pixel using the determinant and trace of matrix $A$.

    $$
    R = \det(\mathbf{A}) - k \cdot (\text{trace}(\mathbf{A}))^2
    $$

    Where:

    $$
    \det(\mathbf{A}) = S_{xx}S_{yy} - S_{xy}^2, \quad \text{trace}(\mathbf{A}) = S_{xx} + S_{yy}
    $$

    And $k$ is an empirically determined formula
4. Apply a threshold to the Harris response to find potential corners. Use non-maximum suppression to ensure that the detected corners are local maxima in a defined neighborhood, which helps eliminate multiple responses to the same corner feature.

The properties of harris corner detector are:

- Locality: The Harris Corner Detector uses local gradients within a window around each point, which makes it sensitive to local image features. The locality is emphasized by the Gaussian weighting in the computation of the second moment matrix $A$.
- Invariance to Illumination: The detector is generally invariant to changes in illumination because it relies on gradient directions rather than intensity values directly. This property allows it to perform well under varying lighting conditions.
- Invariance to Rotation: Since the response $R$ is based on the eigenvalues of $A$, which are invariant to the rotation of the image, the Harris detector can reliably identify corners regardless of image orientation.
- Partial Scale Invariance: While not fully scale-invariant, the Harris Corner Detector can be adapted to detect corners across scales by using a multi-scale approach. This involves applying the detector at different scales using Gaussian blurring and resizing.
- Perspective transformations: More complex geometric transformations, such as skewing (affine transformations) or perspective changes, can alter the relative angles and distances between points in the image, potentially affecting how corners are detected. The Harris detector does not account for such transformations inherently, and corners that are detectable under a frontal view might not be detected under extreme perspective distortions.

##### Harris corner detector scale invariant

To make the Harris Corner Detector robust to scale changes, we can integrate it with scale-normalized Laplacian of Gaussian (LoG) filters.
This approach combines the Harris Detector’s sensitivity to corners with the scale-invariance provided by the LoG filter.
The Laplacian of Gaussian (LoG) is a combination of Gaussian blurring followed by the application of the Laplacian filter, but its use and purpose, especially in conjunction with scale adjustments in feature detection like in the Harris Corner Detector, serve a specific and strategic role.

1. The image is first smoothed using a Gaussian filter. This step reduces noise and blurs the image, which is crucial for reducing the sensitivity of the subsequent Laplacian to small, irrelevant details or noise.
2.After smoothing, the Laplacian filter, which calculates the second derivative of the image, is applied. This highlights regions of rapid intensity change, which are indicative of edges. When applied to the smoothed image, it detects blobs or regions where the intensity changes concavely or convexly, which are characteristic of points like corners.

The Gaussian blur’s scale $\sigma$ is varied to detect features at different scales, while keeping the image size fixed.
When you fix the image size and vary $\sigma$, you are essentially simulating the effect of zooming in and out on the physical scene without actually resizing the image.
A larger $\sigma$ produces more blur, which simulates viewing the scene from a greater distance or at a lower resolution. Conversely, a smaller $\sigma$ retains more of the original detail, representing a closer or higher-resolution view.

The sensitivity of the Laplacian to features of different sizes is adjusted by the preceding Gaussian blur.
At larger scales (higher $\sigma$), the Laplacian responds primarily to large-scale structures.
At smaller scales (lower $\sigma$), it detects finer structures.

This allows the detection of features that are significant at various sizes, which is vital for scale-invariance.
The scale-normalized LoG, helps maintain consistent detection strength across these scales by compensating for the natural weakening of the Laplacian response at larger scales.

The steps are the following:

1. Apply the gaussian filter with varyin $\sigma$ values to create a set of scaled images.
2. For each scaled image, compute the image gradients and determine the second moment matrix.
3. Calculate the Harris response
4. Apply a threshold to the Harris response to determine potential corner points. Additionally, implement non-maximum suppression across both spatial and scale dimensions to ensure that only the most prominent corners are selected. A point is considered a valid corner if its Harris response is a local maximum in the immediate spatial neighborhood and across the scales.

Recalculating the Harris response at each scale allows the detector to adapt to the intrinsic scale of features within the image.
This is crucial for detecting corners that may only be visible or relevant at certain scales.
By examining corners across scales, the method ensures that detected corners are not artifacts of a particular resolution or noise level but are instead significant structural features of the image.

##### Blob detector

A "blob" in the context of image processing and computer vision refers to a region within an image that differs in properties, such as brightness or color, compared to surrounding areas.
These regions are typically characterized by some uniformity within the blob itself and are distinct from the background or other objects in the image.
Blobs are essentially compact clusters of pixels that share common visual characteristics.
Using the Laplacian of Gaussian (LoG) as a blob detector is a classic method in computer vision for detecting regions in images that are significantly different from their surroundings in terms of brightness or color.
After applying LoG,  the next step is to identify the zero crossings in the response.
These are locations in the image where the response changes from positive to negative, indicating potential blob boundaries.
Within the regions bounded by zero crossings, locate the minima of the LoG response.
These minima (negative peaks) are typically located at the centers of blobs.

We can enhache this procedure by using Gaussian pyramids.
At each level of the Gaussian pyramid, apply the Laplacian of Gaussian filter.
Because each level of the pyramid represents the image at a different scale due to downsampling and blurring, applying the LoG at each level allows you to capture blob-like features that are appropriate for that scale.
Evaluate the significance of detected blobs at each scale.
Blobs that appear consistently across scales or that are particularly prominent at a specific scale can be identified as significant features.
For each detected blob, you may also choose the scale at which it appears most prominently as its characteristic scale.
This can be important for applications where understanding the scale of features is necessary.

### Feature descriptors

Feature detectors identify points or regions of interest within an image.
These can be corners, edges, blobs, or other significant local structures.
The main goal is to detect parts of the image that are distinct and invariant under various transformations such as rotation, scale changes, and lighting variations. Detectors do not describe the characteristics of these features; they merely locate them.

Feature descriptors are numerical representations of image characteristics used to quantify and describe unique aspects of an image.
Typically extracted from specific regions or points of interest within an image, these descriptors are designed to be robust to variations in illumination, orientation, and scale.
The goal of a feature descriptor is to provide a concise, yet descriptive, representation of an image region that can be easily compared with other descriptors.

Once features have been detected, feature descriptors then describe these features in a robust and informative way.
The descriptors encapsulate the information about the appearance and shape of the features within their neighborhood.

#### Building HoG

We can introduce the HoG feature descriptor as a series of enhancements to address specific limitations to easier approaches.

##### Image patches

Starting with the image patch, we could directly use the pixel intensity values inside that patch.
A limitation is that this method is sensitive to absolute intensity values and does not account for geometric or photometric transformations.
A matching is only effective under the condition that geometry and appearance remain unchanged, which is often not the case in practical scenarios.

##### Image gradients

We could enhache by transitioninng from using raw intensity values to image gradients.
The rationale being that gradients measure changes in intensity and are more robust against variations in absolute brightness, addressing some of the illumination sensitivity.
While raw pixel values are generally not used alone due to their sensitivity to various transformations, gradients form a foundational element of more complex descriptors because they provide robustness against changes in lighting.

A limitation is that radients measure changes in intensity and are more robust against variations in absolute brightness, addressing some of the illumination sensitivity.

##### Color Histograms

We could use histograms to describe patches.
The rationale being that histograms are invariant to scale and rotation, making them more robust than raw color values.
Color histograms introduce robustness against scale and rotation changes and are generally considered in addition to gradient information.

A limitation is that color histograms lack spatial information, meaning they don’t encode where within the patch specific colors are located, potentially leading to poor feature matching when spatial arrangement is crucial.

##### Spatial Histograms

Instead of replacing color histograms, spatial histograms extend them by incorporating spatial layout.
This method divides the image into cells and computes histograms for each cell, thus maintaining some spatial information.
Typically, spatial histograms use gradient information within each cell to capture both the appearance and the arrangement of features within the patch.
They do not necessarily have to use color information but can if color is a relevant feature for the task.
This approach retains some spatial layout information, providing a balance between color/gradient information and their arrangement within the patch.
A limitation is that while this method includes some spatial information, it’s not completely invariant to rotation, and pixel colors still vary with illumination.

##### Orientation Normalization

We can make the feature descriptor invariant to rotation by aligning the histograms according to a dominant orientation.
A limitation is that even with orientation normalization, issues like the color constancy problem and the non-identifiability of objects solely by color distribution remain.

##### HoG

<!-- https://github.com/ThuraAung1601/human-detection-hog?tab=readme-ov-file -->
HoG, Histogram of Gradients combines the concept of gradients, spatial structuring (similar to spatial histograms), and orientation normalization into a single coherent descriptor.
In HOG, gradients are calculated for each pixel, histograms of these gradients are compiled for spatially defined cells, these cells are grouped into larger, overlapping blocks for normalization, and orientation is considered to ensure rotation invariance.
The underlying principle of HOG is that the appearance and shape of a local object within an image can be described by the distribution of intensity gradients or edge directions.

The HoG algorithm is the following:

1. Preprocessing: Convert the image to grayscale to reduce computational complexity and focus on structural content rather than color.
2. Gradient computation: Calculate the gradient values for each pixel in the grayscale image, focusing on the horizontal (x-direction) and vertical (y-direction) gradients. Typically a sobel filter is used.
3. Orientation binning: Divide the image into small, spatial regions called cells (e.g., 8x8 or 16x16 pixels per cell). For each cell, create a histogram of gradient orientations. Each pixel within the cell contributes to the histogram based on its gradient orientation and magnitude. Typically, orientation angles are binned into 9 to 18 bins covering 0 to 180 degrees (unsigned gradients) or 0 to 360 degrees (signed gradients), depending on whether symmetry is considered.
4. Descriptor blocks: Group adjacent cells into larger blocks (e.g 2x2). Normalize the histograms within each block to counteract effects of varying lighting and shadow within the image. Common normalization schemes include: L2-Norm, L1-norm and L1-sqrt.
5. Concatenate Histograms: Combine the normalized histograms from all the blocks into a single feature vector. This vector represents the HOG descriptor for the entire image or detection window.
6. Sliding Window: For object detection tasks, apply the HOG descriptor within a sliding window that moves across and scales over the image.

HoG has been has been particularly successful and popularized through its application in human detection.
The way HOG captures edge and gradient information is particularly well-suited to the vertical and horizontal lines typical in human forms, making it effective for identifying people in images.
It’s robust against changes in illumination and pose to a certain extent, which are common challenges in human detection scenarios.
In particular we can modify the original HoG procedure to better suite this task:

- 1:2 Aspect Ratio: This ratio effectively captures the vertical orientation of a human body, making it ideal for recognizing standing or walking figures, as it aligns well with the human torso and leg regions.
- 64x128 Pixels: This specific resolution strikes a balance between detail and computational efficiency. It's high enough to capture necessary details for distinguishing human forms but low enough to maintain speed in processing, which is crucial for real-time applications like surveillance.
