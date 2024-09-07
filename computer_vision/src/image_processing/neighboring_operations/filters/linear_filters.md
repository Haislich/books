# Linear Filters

Linear Filters apply a linear operation to the pixels in a neighborhood defined by a kernel or a mask.
The new value of a target pixel is computed as a weighted sum of the pixel values within the kernel's footprint.

Linear filters are typically applied using a process called convolution, which is closely related to another operation known as cross-correlation.
Both convolution and cross-correlation involve sliding a kernel (or filter) over the image and computing the sum of element-wise products at each position.
The subtle difference between these two operations primarily lies in the orientation of the kernel:

- Cross-Correlation: In cross-correlation, the kernel is slid over the image as it is, without flipping. This means the top-left value of the kernel always multiplies the top-left value of the image region it covers.

- Convolution: Convolution involves flipping the kernel both horizontally and vertically before sliding it over the image. This means that the operation is a more complex form of template matching, as it takes into account the spatial orientation of the kernel's weights relative to the target feature in the image.

Convolution is used in signal processing and image processing because it mathematically models the way physical systems respond to stimuli.
In practical image processing, especially with symmetric kernels (like Gaussian blurs), convolution and cross-correlation can produce the same results because the flipping of a symmetric kernel does not change its layout.

Convolution is a key operation used in convolutional neural networks (CNNs), including famous architectures like AlexNet.
CNNs fundamentally use a sequence of learned linear filters (convolutional layers) to process the input images.
This operation allows CNNs to effectively learn features from image data, making them particularly well-suited for tasks like image classification, object detection, and more.
Convolutional layers in a CNN use learned filters (kernels) to convolve across the input image or feature maps from previous layers.
Each filter is designed (or learned during training) to detect specific features such as edges, textures, or more complex patterns depending on the depth in the network.
As the input data passes through successive convolutional layers, the network can form a hierarchy of features from simple to complex.
Lower layers typically detect simple features (e.g., edges and corners), while deeper layers combine these features to detect higher-level structures (e.g., parts of objects or entire objects).

In the context of convolutional neural networks (CNNs), padding and stride are crucial parameters that influence how the convolution operation is applied to the input image or feature maps. Both settings help control the spatial dimensions of the output feature maps and can have a significant impact on the network's performance and efficiency.

Padding refers to the practice of adding layers of zeros (or other values) around the edge of the input image or feature map before applying the convolution operation.
Without padding, the spatial dimensions of the output feature map are reduced with each convolutional layer (unless a stride of 1 is used and the kernel size is 1x1).
Padding allows the convolution operation to cover the bordering elements of the input, enabling the output feature map to maintain the same size as the input.
This is particularly important in deep networks, where many convolutional layers would otherwise progressively shrink the spatial dimensions of the feature maps to a point where no further convolutions could be applied.
Without padding, pixels on the border of the image are used less frequently than pixels in the center when applying filters.
Padding increases the number of times edge pixels are used in convolution computations, helping the model learn from the entire image more effectively.

Stride dictates the number of pixels the convolution filter moves across the input image or feature map after each operation.
A stride greater than one reduces the spatial dimensions of the output feature map.
This is because the filter skips over pixels as it slides across the input.
Using a stride of 2, for example, typically reduces the dimensions of the output feature map to half those of the input, assuming other settings remain constant.
Increasing the stride reduces the computational load and the size of the output, which can speed up the training and inference processes.
A higher stride increases the effective field of view of each application of the convolution kernel, allowing it to cover a broader area of the input with fewer operations.
This can help the network capture more global features faster, though it may reduce the granularity of the feature maps.

## Gaussian Filters

Gaussian filters are a type of image smoothing filter that reduce noise and detail in images using a Gaussian function.
They are characterized by their bell-shaped curve in one dimension and by a surface whose sections are Gaussian curves in two dimensions.
Gaussian filters are widely used due to their properties in the spatial and frequency domains.
Due to its nature, the Gaussian filter provides smooth gradients without sharp transitions, making it ideal for blurring and for use in scale-space representation.
In two dimensions, a Gaussian filter is circularly symmetric (isotropic), meaning it blurs uniformly in all directions.
Gaussian filters have a low-pass characteristic, which means they attenuate high-frequency components more than low-frequency components, effectively reducing image noise and detail.

A separable filter is a type of filter that can be broken down into the product of two or more one-dimensional kernels.
This allows the two-dimensional convolution operation to be performed more efficiently by reducing it to multiple one-dimensional convolutions.

Gaussian filters are inherently separable, which is one of their advantageous properties.
A two-dimensional Gaussian kernel can be expressed as the outer product of two identical one-dimensional Gaussian kernels.
This means that a 2D Gaussian blurring operation can be efficiently implemented by first applying a 1D Gaussian blur vertically across the image and then horizontally.
This separability makes Gaussian filters particularly appealing for real-time processing and applications in computer vision where computational resources are a concern.

## Sharpen Filter

Sharpening filters are used in image processing to enhance the visibility of edges and fine details in images.
They work by emphasizing high-frequency components, which correspond to rapid changes in image intensity, such as edges.
This is typically achieved through a process that enhances the contrast at these high-frequency locations.

A common approach involves using a kernel that accentuates edges.
This can be done with kernels that approximate the second derivative of the image (like the Laplacian filter), or by simply using a kernel that boosts center pixel intensity while subtracting a fraction of the neighboring pixel intensities.

Gaussian filters are inherently smoothing filters and are typically used to reduce noise and detail in the images.
When used as part of an unsharp masking technique, the Gaussian filter serves to isolate the low-frequency components by smoothing out the high frequencies.
The subtractive process in unsharp masking that follows the Gaussian blurring emphasizes the high-frequency components (edges) by reducing the weight of low frequencies.
This process effectively increases the sharpness of the image.
