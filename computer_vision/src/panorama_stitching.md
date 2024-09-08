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

#### SIFT

Scale-Invariant Feature Transform (SIFT) is both a feature detector and a descriptor.
As a detector, SIFT identifies key points in an image that are invariant to scale and orientation changes, and somewhat robust to changes in illumination and noise.
These key points are typically distinct and localized in the image, making SIFT particularly effective for tasks like object recognition and registration where matching specific features across different images is crucial.

The steps of the algorithm are:

1. Scale-Space extrema detection: The first step in SIFT is to construct a scale space, which is typically achieved by convolving the image with Gaussian filters at different scales. This results in a set of blurred images each representing the input image at a different scale. Once the scale space is constructed, the Difference of Gaussians (Difference of Gaussians is a more efficent approximation of LoG) is computed by subtracting consecutive Gaussian-blurred images within the same octave of the scale space. These DoG images are used to find potential keypoint locations.
2. Keypoint localization: In this step, each pixel in the DoG images is compared with its 8 neighbors in the same image and 9 neighbors in the scale above and below (26 neighbors in total). Pixels that are local maxima or minima are selected as candidate keypoints. The positions of these candidate keypoints are further refined to achieve sub-pixel accuracy by using a Taylor expansion of the scale-space function. Keypoints with low contrast or poorly localized along edges are discarded. This is done by calculating the Laplacian or using a Hessian matrix and eliminating keypoints that do not meet certain stability criteria.
3. Keypoint orientation: For each image sample in a region around the keypoint, the gradient magnitude and orientation are calculated. The gradients are weighted by a Gaussian window centered at the keypoint location. This Gaussian window decreases the influence of gradients that are farther away from the keypoint, focusing more on those in the immediate vicinity, which are more likely to be relevant to the actual keypoint. An orientation histogram is created from the gradient orientations of the sampled points within the region, using the weighted gradients. Each sample adds to a bin in the histogram based on its orientation, with the contribution weighted by its gradient magnitude and the Gaussian weight. The histogram typically has 36 bins covering 360 degrees. The peaks in the histogram represent the most prominent orientations within the sampled region. The highest peak is selected, and any other peak that is within 80% of the highest peak’s magnitude is also considered. This approach allows for the keypoint to have multiple orientations, which can be useful for matching keypoints under different rotations.

Once the feature are found, they are described as follow:

1. Each keypoint is associated with a specific scale, determined during the detection phase. The image used for the descriptor calculation is the one that corresponds to this scale in the Gaussian pyramid. This ensures that the local characteristics around the keypoint are appropriately represented for that scale.
2. Gradients (both magnitude and direction) are calculated for each pixel within a 16x16 window centered on the keypoint. These gradients provide the fundamental components used to describe the pattern of intensities around the keypoint. Gradients are sensitive to edges, corners, and other texture elements that are crucial for describing the local area uniquely.
3. To achieve rotation invariance, the gradients calculated in step 2 are adjusted according to the keypoint’s orientation. This means rotating the coordinate system of the 16x16 window so that the orientation of the keypoint becomes the new 'up' direction in the window. This rotation ensures that the descriptor is consistent even if the image is rotated.
4. The large 16x16 window is subdivided into smaller 4x4 sub-blocks or cells. This subdivision allows the descriptor to capture more localized information within the window, enhancing the ability to handle changes in local appearance and structure.
5. Within each 4x4 cell, an orientation histogram is computed. The gradient magnitudes contribute to the histogram bins based on their orientation, with the contribution weighted by the magnitude of each gradient. Additionally, these contributions are typically weighted by a Gaussian window centered on the keypoint to give higher importance to gradients closer to the center of the window. This results in each cell capturing a summarized but detailed representation of the gradient directions within that cell.
6. The orientation histograms from all 16 cells (4x4 grid) are concatenated to form the final SIFT descriptor. Since each histogram consists of 8 bins and there are 16 cells, the resulting descriptor is a 128-element vector. This vector encodes the local texture and gradient patterns around the keypoint in a way that is distinctive and robust to changes in viewpoint, scale, and illumination.

The SIFT (Scale-Invariant Feature Transform) descriptor is widely appreciated for its robust properties, which make it highly effective in computer vision tasks like object recognition, image matching, and 3D reconstruction.

- Invariance to scale: The SIFT descriptor is designed to be invariant to changes in scale. Features are detected at multiple scales in the scale space, and each keypoint is described in terms of its relative scale, ensuring that the same feature can be recognized at different sizes.
- Invariance to Rotation: Orientation normalization during the descriptor computation process ensures that the descriptor is invariant to image rotation. By aligning the gradient orientations relative to the keypoint orientation, the descriptor maintains consistency regardless of how the object is oriented in the scene.
- Robustness to Affine Transformation: While primarily designed for scale and rotation invariance, SIFT descriptors are also robust to a degree of affine transformations, which include translation, scaling, rotation, and shearing. This robustness is crucial for matching features across images that have been taken from different viewpoints.
- Partial Invariance to Illumination Changes: The SIFT descriptor is somewhat invariant to changes in illumination. The normalization step in the descriptor formation reduces the effects of varying lighting conditions by normalizing the vector to unit length, thus making the descriptor more about the relative gradients than their absolute values.
- Distinctiveness: Each SIFT descriptor is highly distinctive, which is achieved by the detailed 128-element vector that captures substantial local gradient details within a region. This distinctiveness allows for reliable matching of keypoints between different images, even among thousands of keypoints.
- Robustness to Noise: Gaussian blurring used during the detection and description phases of SIFT helps in reducing the effect of noise in the image. The descriptor's reliance on gradient orientations and magnitudes, rather than raw pixel intensities, further helps mitigate the impact of noise.

#### SURF

The Speeded Up Robust Features (SURF) algorithm is known for being faster and somewhat more robust than SIFT, due to some simplifications and optimizations in its approach.
SURF, like SIFT, is used for detecting keypoints and extracting feature descriptors, but it has been specifically designed to increase the speed of the process while maintaining robustness.

During the interest point detection SURF relies on the integral image concept, use of Hessian matrix determinant for keypoint detection, and involves several steps that make it particularly effective and efficient:

- Integral image: The first step in SURF is the computation of the integral image (also known as a summed-area table). An integral image allows for rapid summation of pixel values within a rectangular area and is used to compute box-type convolutions quickly, which approximate Gaussian convolutions. This transformation facilitates the rapid calculation of image features at different scales and makes the algorithm much faster compared to techniques that rely on repeated filtering.
    $$
    \text{Integral Image}(x, y) = \sum_{i \leq x, j \leq y} \text{Image}(i, j)
    $$
- Hessian matrix determinant for keypoint localization: the Hessian matrix is used at different scales in scale-space to determine possible interest points.
    $$
    H(x, \sigma) = \begin{pmatrix}
    L_{xx}(x, \sigma) & L_{xy}(x, \sigma) \\
    L_{xy}(x, \sigma) & L_{yy}(x, \sigma)
    \end{pmatrix}
    $$
    The Hessian matrix is a crucial tool in SURF and other image processing algorithms because it provides a concise way to capture the second-order local image curvature. This information is essential for determining potential interest points or keypoints in images.
    Here $L_{xx}$ , $L_{yy}$ and $L_{xy}$ are the second order derivative gaussian derivatives of the image w.r.t x, y and mixed.
    To identify keypoints, SURF calculates the determinant of the hessian matrix at each pixel and scale
    $$
    \text{det}(H) = L_{xx}(x, \sigma) \cdot L_{yy}(x, \sigma) - (L_{xy}(x, \sigma))^2
    $$
    The determinant of the Hessian matrix is sensitive to blob-like structures in the image where the curvature is either concave or convex in both dimensions. This property makes it particularly useful for identifying points of interest that are distinct and stable, which are good candidates for keypoints.By calculating the Hessian matrix at various scales, SURF efficiently analyzes the scale-space of the image. This scale-invariant feature detection ensures that features can be recognized at different sizes, which is crucial for many computer vision applications. The locations of local maxima and minima of the determinant across scale and space pinpoint potential keypoints. These points are where the determinant values are higher or lower than all the neighboring points in the scale-space, indicating significant local features.
    Determining the exact Hessian matrix using second-order Gaussian derivatives is computationally expensive, especially when applied to every pixel of an image across multiple scales. SURF optimizes this process using box filters, which are a type of integral filter that approximates the Gaussian second derivatives.
- Once the determinants are computed across the scale space, SURF looks for local maxima and minima in this determinant map. This is done in a non-maximum suppression step, where each point is compared to its neighbors in the same scale as well as the scales above and below.
Points that are local maxima or minima represent potential keypoints; these points indicate areas where the image structures show significant blob-like structures, which are distinctive and stable for feature matching.
- Detected keypoints are then filtered based on the determinant value. Points with a determinant below a certain threshold are discarded to eliminate less stable and less distinctive points. Further refinement involves interpolating the scale and location to achieve subpixel and sub-scale accuracy. This step enhances the precision of keypoint localization, ensuring that the keypoints are accurately positioned relative to their true locations in the continuous image domain.
- To ensure that the keypoints are not located along edges (which are less stable for matching), SURF applies a measure involving the trace and determinant of the Hessian matrix. Keypoints for which the ratio of the determinant to the trace (or similar measures) indicates an edge response are discarded.

After keypoints are localized and assigned an orientation, the next step is to build a robust descriptor that effectively captures the local image structure around each keypoint.
This descriptor can then be used for matching keypoints between different images.
The descriptor phase in SURF involves several steps to ensure that the descriptor is both descriptive and robust to various transformations:

- Before forming the descriptor, each keypoint is assigned a dominant orientation based on the sum of Haar wavelet responses within a circular neighborhood. This orientation is used to ensure rotation invariance by aligning the descriptor relative to this direction.
- A square region centered on the keypoint and aligned with the assigned orientation is defined. The size of this region is scaled according to the scale at which the keypoint was detected, ensuring scale invariance.
- This square region is divided into smaller 4x4 subregions. This subdivision allows the descriptor to capture fine details within the neighborhood while maintaining a robustness to small deformations and local geometric distortions.
- For each of the 4x4 subregions, Haar wavelet responses are computed both in the direction of the orientation and perpendicular to it These responses are calculated over a 2x2 sub-subregion within each 4x4 block to efficiently cover the entire neighborhood.
- For each 4x4 subregion, the following features are computed and compiled into the descriptor vector:
  - The sum of the Haar wavelet responses $dx$ and $dy$
  - The sum of the absolute values of the Haar wavelet responses $|dx|$ and $|dy|$.
    These sums are calculated to capture both the polarity and the magnitude of the gradient information, which are crucial for describing the texture and structure of the local area.
- The descriptor for each keypoint consists of the summed responses for each of the 4x4 subregions. 4 sums for each subregion. Given 16 subregions, the descriptor has a total of 64 elements (4 features per subregion × 16 subregions).
- Finally, to achieve robustness against variations in illumination and contrast, the descriptor vector is normalized to unit length. This normalization step ensures that the descriptor's effectiveness is not influenced by changes in lighting conditions.

In SURF, the matching procedure between feature descriptors extracted from different images is a critical step for applications like object recognition, image stitching, and 3D reconstruction.
The efficiency and accuracy of this matching process are significantly enhanced by techniques such as Laplacian indexing.

- Once descriptors have been extracted from images, the next step is to find corresponding descriptors between two images. This is usually done using a distance metric, most commonly the Euclidean distance. Each descriptor in one image is compared to descriptors in the other image to find the closest match.
- The simplest and most common approach to matching descriptors is the nearest neighbor search, where each descriptor from one set is matched with the descriptor in the other set that has the smallest Euclidean distance.
- To improve the robustness of the matching and reduce false matches, SURF often employs the ratio test proposed by Lowe in the context of SIFT. In this test, the distance to the nearest neighbor is compared to the distance to the second-nearest neighbor. A match is considered good if the ratio of these distances is below a certain threshold

Laplacian indexing is an additional step used in SURF to enhance the matching process by considering the sign of the Laplacian (the trace of the Hessian matrix) at the location of keypoints.

- Positive Laplacian: Indicates a dark blob on a light background (the Laplacian is positive at dark regions surrounded by lighter pixels).
- Negative Laplacian: Indicates a light blob on a dark background (the Laplacian is negative at bright regions surrounded by darker pixels).

Before comparing descriptors, Laplacian indexing can be used to filter out pairs of keypoints between images that do not have the same type of contrast polarity (i.e., both keypoints should either be dark blobs on light backgrounds or light blobs on dark backgrounds).
This preliminary check reduces the computational load by avoiding unnecessary comparisons and also increases the likelihood that the matches are correct.
By ensuring that only keypoints with the same contrast polarity are compared, Laplacian indexing speeds up the matching process and improves the accuracy by reducing false positives.

#### Binary descriptors

Binary descriptors play a crucial role in computer vision, particularly in applications where speed and efficiency are paramount.
They transform the way that keypoint descriptions are stored and compared, offering several advantages over traditional feature descriptors.

##### BRIEF

BRIEF is a feature descriptor that generates a binary string representation of an image patch by comparing the intensities at pairs of pixels within the patch.
It involves selecting a set of location pairs within a smoothed image patch and comparing their intensities.
The result of each comparison (whether the intensity at one location is greater than the other) contributes one bit to the final binary descriptor.
The descriptor is therefore very fast to compute and compact, but it is not rotation invariant and somewhat sensitive to noise.

##### ORB

ORB is essentially a combination of FAST (Features from Accelerated Segment Test) keypoint detector and BRIEF descriptor with modifications to enhance performance.
It was designed to be a free alternative to SIFT and SURF, being both efficient and effective.
ORB introduces a mechanism to add orientation to the keypoints to achieve rotation invariance.
It also uses a learning-based method to select the most informative and robust pairs of pixels (from a set trained on a corpus of images) for the binary comparison in BRIEF.

##### FREAK

FREAK is inspired by the human visual system, particularly the pattern of retinal ganglion cells, which are denser towards the fovea and become sparser towards the periphery.
FREAK generates binary descriptors by comparing image intensities across a retinal sampling pattern.
The sampling pattern uses overlapping receptive fields that increase in size in a logarithmic manner from the center towards the periphery, which mimics the distribution of human retinal cells.
It is more robust to rotation and scale changes compared to BRIEF and has a lower computational cost compared to more complex descriptors like SIFT or SURF.

### Feature Matching

Feature matching involves identifying corresponding features (or keypoints) between different sets of images.
This process is crucial for many applications, such as stereo vision, object recognition, motion tracking, image stitching, and 3D reconstruction.
Feature matching aims to establish correspondences between sets of features extracted from different images.
These features are often described by descriptors, which encapsulate key information about the feature and its surrounding area in a compact form.

- Feature detection: First, features or keypoints in each image are detected. These keypoints are typically points of interest that can be reliably identified and are invariant to changes in scale, orientation, and lighting.
- Feature description: Each detected feature is then described by a feature descriptor.The descriptor encodes the appearance of the feature in a way that is intended to be distinctive and robust against transformations. The goal is for the same feature to have similar descriptors even when captured under different conditions.
- Once features are described, the next step is to match these descriptors between different images. This involves finding pairs of descriptors that are closest according to some distance metric (like Euclidean distance for real-valued descriptors or Hamming distance for binary descriptors).
- The choice of distance metric depends on the type of descriptor used. Real-valued descriptors (e.g., from SIFT or SURF) are typically matched using Euclidean distance, while binary descriptors (e.g., from ORB or BRIEF) use Hamming distance because it is computationally more efficient.

### Deep Learning Approaches

Modern approaches uses Deep learning.
These techniques represent significant advancements in the way local features are detected, described, and matched across images.

#### SuperPoint and SuperGlue

SuperPoint consists of a single convolutional neural network (CNN) that simultaneously learns to detect interest points and describe them.
The model is initially trained on synthetic data and then fine-tuned through self-supervised or semi-supervised learning on real-world images.

- Dual Head Architecture: SuperPoint utilizes a CNN with a shared encoder and two separate heads: one for interest point detection and another for descriptor computation. The shared encoder processes the input image and extracts feature maps that are common to both tasks, which is efficient and leverages shared information.
- This head processes the output of the shared encoder to produce a dense heatmap where each pixel value represents the probability of being an interest point. Training this part involves using a ground truth heatmap where known keypoints are marked.
The keypoints are then typically selected by choosing local maxima in the heatmap, often with non-maximum suppression to ensure that the keypoints are well-distributed and non-overlapping.
- The descriptor head outputs a dense descriptor map where each pixel has a corresponding feature vector (descriptor). This map has the same spatial dimensions as the heatmap but with a deeper channel dimension representing the descriptor length.
Descriptors are extracted directly from locations corresponding to detected keypoints. These descriptors encode the local appearance of the image around each keypoint and are designed to be distinctive and robust against various image transformations.

SuperGlue is a novel method for matching features across different images, which complements SuperPoint by providing a powerful matching mechanism.
It uses a Graph Neural Network (GNN) to consider the relationships between features in each image as well as across images.
SuperGlue is designed to address the limitations of traditional feature matching techniques which often rely solely on nearest neighbor matching and may struggle with complex scenarios involving significant changes in viewpoint, scale, or occlusions.
SuperGlue, through its graph-based approach, aims to maximize the correctness of matches by considering the relationships and spatial arrangements between features across images.

As previously discussed, SuperPoint serves as the feature detection and description method. It provides keypoints and associated descriptors that are robust and repeatable across different views of the same scene.
`The keypoints and descriptors generated by SuperPoint are fed into SuperGlue as the foundational elements for matching.

- Graph Construction: SuperGlue constructs a graph where nodes represent the feature descriptors (from SuperPoint or any other feature detector), and edges represent possible matches. The graph includes nodes for features from both of the images that are being matched.
- Graph Neural Network: The GNN processes this graph to refine the potential matches by considering both local evidence (how well the descriptors match) and global context (the geometric arrangement of features and consistency of potential matches across the graph).
- Match Filtering: SuperGlue uses the output from the GNN to select matches that are both locally and globally consistent, significantly improving the accuracy and robustness of the matching process compared to traditional methods that often rely solely on nearest neighbor searches and simple geometric constraints.

## Image transformations

In the context of computer vision, a transformation refers to the application of a mathematical operation to modify the geometry of an image.
This can include changes to the position, size, orientation, or perspective of the image.

A transformation in computer vision is a function that maps a set of points from one image to a set of corresponding points in another image or within the same image.
It essentially modifies the spatial relationships between points.

Linear transformations include operations like translation, rotation, scaling, and shearing.
These transformations can be represented as linear equations, making them particularly straightforward to implement and compute.

Homogeneous coordinates are used to represent transformations that involve perspective changes or more complex manipulations that cannot be achieved through linear transformations alone. These are crucial for achieving what’s known as projective transformations or homographies.

Homogeneous coordinates introduce an additional dimension (making points from 2D go to 3D, or 3D go to 4D).
This allows for the representation of infinity and the ability to perform all affine transformations, including translations, as matrix multiplications.
In simpler terms, they facilitate the representation of points at infinity by normalizing with respect to the additional coordinate, which is useful for perspective transformations where parallel lines appear to converge at infinity.

In the context of homogeneous coordinates, "points at infinity" are a key concept that allows for the representation and manipulation of geometrical transformations in a more comprehensive way, particularly when dealing with perspectives in images.
Points at infinity are used to handle directions or locations that are conceptually "infinitely far away" but still affect the geometry, like the vanishing points in perspective drawings.
In homogeneous coordinates, a point at infinity is represented by setting $w=0$.
These representations are essential in projective transformations where parallel lines in the real world appear to converge at infinity, such as in perspective projections.

This representation is especially important in tasks like panorama stitching, where images may need to be warped in complex ways to align seamlessly due to differences in camera orientation and position.

### Linear transformation

Linear transformations include operations such as rotation, scaling, and shearing.
They are linear because they adhere to the superposition principle and are homogenous w.r.t scalar multiplication.

These transformations can be represented by multiplying a coordinate vector by a matrix, and they maintain the linear relationships and distances between points.

### Affine transformations

Affine transformations are a bit more general than linear transformations.
They include all linear transformations plus translation.
Affine transformations can be represented in matrix form when using homogeneous coordinates, which allows the translation component to be embedded in the matrix operation.
They satisfy the property that parallel lines remain parallel after the transformation.
The general form of a affine transformation:

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

Projective transformations, or homographies, go further than affine transformations.
They can transform lines into lines but do not necessarily preserve parallelism, making them capable of handling perspective changes.
Homographies are not linear due to their ability to map points to and from infinity, which disrupts the superposition principle fundamental to linear transformations.
Homographies can warp, tilt, and change the scale of the image content in ways that simulate a change in viewpoint, as seen through a camera lens.
Homographies are represented by a 3x3 matrix, which can transform homogeneous coordinates of points in one image to those in another.
The general form of a homography matrix:

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

### Computing transformations

Computing the transformation involves determining the transformation matrix $T$  that transformed image $A$ into image $B$, given a set of matched feature points between the two images.
It is assumed that the number of matches is sufficiently large to reliably compute the transformation.

#### Computing translation

Given matched feature points between two images, you want to compute the translation vector $(t_x,t_y)$ that best maps the matched points in $A$ into the matched points in $B$.
For each matched point $(x_i,y_i)$ in A and $(x_i',y_i')$ in B, the translation equations can be written as:

$$
x'_i = x_i + t_x \\
y'_i = y_i + t_y
$$

We could rearrange to isolate the translational components:

$$
t_x = x'_i - x_i \\
t_y = y'_i - y_i
$$

This formulation can be extended across all matches to create a system of equations.
While theoretically, two matches would be enough to compute $t_x$ and $t_y$, using multiple matches provides a more robust estimate especially in the presence of measurement noise or imperfect feature matching.
The least squares method addresses this by finding a solution that best fits all the data, rather than attempting to exactly solve each individual equation—which may not be feasible due to conflicts among the data points caused by measurement errors. We cannot be 100% confident in the accuracy of each match; hence, least squares is advantageous as it minimizes the sum of squared differences between the observed translations (from the matched features) and the model's predicted translations.

$$
\min_{t_x, t_y} \sum_{i=1}^n ((x'_i - x_i - t_x)^2 + (y'_i - y_i - t_y)^2)
$$

For which the explicit solution is:

$$
\hat{t}_x = \frac{1}{n} \sum_{i=1}^n (x'_i - x_i)
\hat{t}_y = \frac{1}{n} \sum_{i=1}^n (y'_i - y_i)
$$

This formulation effectively "averages" the translation estimates from all matches, reducing the impact of any erroneous data points and improving the reliability of the translation estimate.

#### Computing affine transformations

When extending the concept of estimating transformations from simple translation to more complex affine transformations, the mathematical framework and the estimation technique (like least squares) need to accommodate the additional degrees of freedom that affine transformations introduce.
An affine transformation in two dimensions includes not only translations but also rotations, scaling, and shearing, making it more general than a pure translation.
An affine transformation can be represented by a 2x3 matrix, or more conveniently in homogeneous coordinates, by a 3x3 matrix.

For each mathed pair of feature points $(x_i,y_i)$ in A and $(x_i',y_i')$ in B we obtain:

$$
x'_i = a \cdot x_i + b \cdot y_i + t_x \\
y'_i = c \cdot x_i + d \cdot y_i + t_y
$$

Since each match provides two equations and you have six unknowns to solve for, you need at least three matches to form a system of equations that is exactly determined.
More specifically 3 matches provide 6 equations, which match the number of unknowns.
The cost function in the context of affine transformation estimation is typically defined as the sum of squared differences between the observed transformed points in the second image and the points predicted by the affine transformation applied to the first image.
The residuals are the differences in each dimension (x and y) for each matched point pair.
The cost function can be written as:

$$
\min_{a, b, c, d, t_x, t_y} \sum_{i=1}^n ((x'_i - (a \cdot x_i + b \cdot y_i + t_x))^2 + (y'_i - (c \cdot x_i + d \cdot y_i + t_y))^2)
$$

#### Computing homographies

We can extend this to homographies.
While the relationship between the input and output coordinates under a homography is generally non-linear due to the division by the third coordinate, it can be linearized using homogeneous coordinates.

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix} =
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

This relationship involves the division by the third coordinate for both $x'$ and $y'$, which is non linear:

$$
x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}} \\
y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}
$$

However, this can be linearized by cross-multiplying to eliminate the denominators, leading to a linear form suitable for solving with linear algebra methods:

$$
x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13} \\
y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}
$$

From the linearized equations, each pair of matched points gives two equations, and you need at least four pairs of points (providing eight equations) to solve for the eight unknowns.
We assume that $h_{33}$ is set to 1 to resolve the scale ambiguity of homogeneos coordinates, reducing the number of unknowns to $8$.
When we set $h_{33}$ to 1, we essentially fix the scale of the transformation, which helps in stabilizing the solution.
It prevents the solution from trivially reaching zero or becoming unbounded.

The objective in least squares is to minimize the residual sum of squares between the observed outputs and the outputs predicted by the linear model.
In this case, the goal becomes minimizing:

$$
\|Ah - b\|^2
$$

Where $A$ is the matrix composed by the constrained coefficient of the linearized equations.
To solve this problem a common approach is to use SVD.
SVD is a powerful algebraic tool used in linear algebra for solving systems of linear equations, among other applications.
It is particularly useful in the context of least squares problems because it can decompose any matrix into its constituent components, helping to analyze the properties of the matrix and find the best solutions, even in cases where other methods (like direct inversion) might fail.
SVD breaks down A into three special Matrixes

$$
A = U \Sigma V^T
$$

In an ideal scenario where noise and errors are minimal, the rank of A would be 8, indicating that the homography vector h lies in the null space of A (h atisfies Ah = 0).
In practice A has often rank 9,  due to noise and other imperfections in data. This indicates that there is no exact solution that completely satisfies all the equations (since there is no non-trivial solution that puts h ientirely in the null space of A).
The problem in this case becomes the minimization of error, because no h will perfectly satisfy the initial equation.

The smallest singular value of A is critical as it indicates the least error or residual in fitting the model to the data.
The smaller this value, the better the fit of the homography can be estimated.
The singular value of A can be thought of as the positive square root of an eigenvalue of $A^TA$

$$
A^T A = V \Sigma^2 V^T
$$

The optimal vector h, the one that defines the homography is the eigenvector of $A^TA$ corresponding to its smallest eigenvalue.
This vector minimizes the reconstruction error in the least squares sense.
The use of SVD provides a numerically stable method for solving overconstrained systems and helps ensure that the computed solution is as precise as possible, minimizing the potential for numerical errors.

##### RANSAC

In the process of computing transformations between two images, A and B, we start by identifying and computing features in each image.
These features are subsequently matched across the images, assuming an optimal matching process.
The goal is to find the transformation vector h that best fits these correspondences by minimizing a least squares solution.
However, the accuracy of this solution heavily relies on the quality of the feature matches.

In the realm of matched image features, outliers represent pairs of points that do not correspond to the same physical point in space.
Such discrepancies often arise due to several factors:

- Detection and Matching Errors: Imperfections in feature detection or descriptor matching can lead to incorrect correspondences.
- Similar Features in Different Locations: Identical textures or patterns located in different parts of the scene might be wrongly matched.
- Dynamic Scene Elements: Moving objects between the captures or changes in the scene can introduce mismatches.
- Perspective and Dimensional Differences: Variations in the 3D structure that are not evident in the 2D projections of the images can also contribute to outliers.

These outliers can significantly compromise the estimated homography.
Since the least squares method assumes all matched points are accurate correspondences, outliers can disproportionately skew the computed transformation, resulting in a homography that fails to align the images correctly.

To address the challenge posed by outliers, the Random Sample Consensus (RANSAC) algorithm is employed.
RANSAC is an iterative method designed to estimate the parameters of a mathematical model robustly by identifying and utilizing only the inliers from the dataset.

1. Sample Minimum Points: Initiate by randomly selecting the minimal set of point pairs required to define the homography. Typically, four pairs are needed to compute an exact model.
2. Fit Model: Calculate the homography using these selected points to establish a potential transformation model.
3. Evaluate Inliers: Apply this model to all matched points, determining the inliers based on their conformity to the model within a specified error threshold, typically measured by geometric distance.
4. Iterate for Optimization: Repeat the above steps multiple times, each with a different set of randomly selected point pairs, to explore various potential models.
5. Select Optimal Model: Choose the homography that yields the highest number of inliers, indicating the most reliable and accurate model.
6. Refine the Model: Finally, refine this model by recalculating the homography based on all identified inliers to ensure the most precise alignment.

Given:

- s: Number of samples or minimum sets of point pairs required to estimate the model (e.g., a homography or another transformation). In the context of homographies, you often need at least four pairs to compute the transformation.
- e: Proportion of outliers in the dataset. This is a critical factor because a higher proportion of outliers means a lower chance of randomly selecting a clean set of inliers.
- p: Desired probability of success, meaning the probability that after N iterations of RANSAC, we successfully find a good model. Commonly, we desire p to be very high (e.g., 0.99 or 99% confidence level).

The number of RANSAC iteration is given by:

$$
N = \frac{\log(1 - p)}{\log(1 - (1 - e)^s)}
$$

TODO pros and cons of ransac

## Warping

Once you've computed the correct homography between two images, the next step often involves applying this transformation to align or stitch the images together.
This process is known as warping. Warping transforms all or part of an image to a new coordinate system.
There are two primary types of warping:

- Forward Warping: Forward warping directly applies the homography transformation to each pixel in the source image to find its new position in the destination image. Because the transformation of discrete pixel locations might not align perfectly with the grid of the destination image, this can create gaps ("holes") where no pixels are mapped, or overlaps where multiple source pixels map to the same destination pixel. These issues necessitate the use of filling algorithms or decisions on how to handle overlapping pixels, potentially compromising image quality.
- Inverse Warping:  Inverse warping works by iterating over each pixel in the destination image and uses the inverse of the homography to map these coordinates back to the corresponding coordinates in the source image. Inverse warping eliminates the problem of gaps, as every pixel in the destination image is accounted for. It naturally handles overlaps through the warping process without needing complex decisions on which pixel to choose.

Both forward and inverse warping usually require coordinates to be mapped to non-integer positions.
Since pixel values can only be defined at integer grid points, interpolation is used to estimate pixel values at non-integer positions.

## Blending

Blending is a critical step in panorama stitching to ensure that the joined areas between multiple images appear seamless and visually pleasing.
When individual images are simply stitched together without blending, the result often exhibits visible seams, differences in exposure, color inconsistencies, or abrupt transitions, especially if the images were captured under varying lighting conditions or with different camera settings.

Blending helps to create smooth transitions between overlapping areas of stitched images.
This prevents harsh lines or visible boundaries that can disrupt the visual flow of the panorama.
Different images may have variations in color balance and exposure levels due to different shooting angles and lighting conditions.
Blending techniques can harmonize these differences, making the panorama look as though it was shot in one take.

Common blending techniques are:

- Feathering: It involves creating a soft transition between images by:
  - Using a Weighted Mask: A gradient mask (or feather) is applied across the overlap region. The mask typically has higher transparency (or lower weight) at the edges and increases toward the center.
  - Linearly Interpolating: The final pixel values in the overlap region are computed by linearly interpolating between corresponding pixel values of the overlapping images, weighted by the feather mask. This reduces the visibility of seams but can sometimes lead to ghosting effects if the images are not perfectly aligned.
- Pyramid blending uses a multi-scale approach to combine images:
  - Image Pyramids: Each image is decomposed into a set of layered images (each layer is a reduced-resolution version of the original). This is often done using Gaussian pyramids.
  - Blend Each Layer: Corresponding layers of the image pyramids are blended separately using simple methods like averaging.
  - Reconstruct: The final blended image is reconstructed from the blended pyramid layers. This technique handles variations in image content effectively over different scales, reducing the risk of ghosting and providing more seamless blends.
- Laplacian Pyramid Blending: A more sophisticated form of pyramid blending, Laplacian pyramid blending, specifically uses Laplacian pyramids:
  - Laplacian Pyramids: These are formed by subtracting each level of the Gaussian pyramid from the next level up, focusing on the image details rather than the broader scale.
  - Blend Pyramids: The Laplacian pyramids of the images are blended at each level.
  - Reconstruct from Blend: The final image is reconstructed from the blended Laplacian pyramid. This method is particularly good at preserving edge information, making it very effective for blending images with significant overlap and detail.
