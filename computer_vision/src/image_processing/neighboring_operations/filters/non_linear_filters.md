# Non Linear Filters

Non-Linear Filters involve operations where the output is not a linear combination of the input pixel values. The operation may include conditions, thresholds, or more complex relationships that do not satisfy linearity.

## Median Filters

Median filters are a type of non-linear digital filtering technique, often used to remove noise from an image or signal.
The median filter is particularly effective at removing 'salt and pepper' noise while preserving edges in an image.
The median filter operates by sliding a window (kernel) over each pixel of the image. For each position of the window, the pixel values covered by the window are sorted numerically. The median value of the sorted pixels is then determined, and this median value replaces the pixel value at the center of the window.
Unlike mean filtering, median filtering does not blur the edges, as the median is less sensitive than the mean to extreme values (which are often edge pixels).

## Bilateral Filter

Bilateral filtering is an advanced method of image smoothing that provides edge preservation while reducing noise, addressing a common limitation found in traditional Gaussian filters.

Gaussian filters are excellent for blurring and noise reduction in images because they effectively smooth variations in intensity.
They work by applying a Gaussian kernel to the image, which averages the pixel values in a way that nearby pixels have a larger influence on the output than distant ones.
However, this isotropic smoothing does not discriminate between edges and noise; it blurs everything indiscriminately, including important edge details.
As a result, while Gaussian filters reduce noise, they also tend to blur sharp edges, which can be undesirable in applications where maintaining edge clarity is important.

Bilateral filtering was introduced to overcome the edge-blurring problem of Gaussian filters.
It does this by taking both spatial proximity and the intensity similarity into account when performing the smoothing:

- Spatial Component: Like Gaussian filtering, bilateral filtering considers the closeness of pixels. This is typically modeled using a Gaussian distribution that decreases the weights of pixels based on their spatial distance from the target pixel.

- Range Component: Unlike Gaussian filtering, bilateral filtering also considers the similarity in intensity values. This range filter ensures that only pixels with intensity values similar to the target pixel are considered for averaging. This component is also typically modeled using a Gaussian distribution, but instead of spatial distance, it uses the intensity difference.
