# Non linear pointwise operations

## Gamma corection

Gamma correction is a non linear transform that is often applied to images before further processing.

$$
p_{out} = p_{in}^{\gamma}
$$

Where $\gamma \approx 2.2$, which is in general not a good approximation but works well in practice.
Gamma correction is used to align digital images to human visual perception, improving image quality and giving the image consistency accross different devices.

## Histrogram equalization

The histogram of each chanel and luminance values of an image describes the set of intensity values in an image.
From the histogram of an image we can compute relevan statistics for the image, in particular we can determine the pixel intensity distribution.

Histogram equalization is a technique used to automatically determine the best contrast values in an image by adjusting the distribution of pixel instensity values of an image such that the histogram of pixel instensities is more uniform across the possible instisity values.

Histogram equalization is a global operation because it considers the distribution of pixel intensities across the entire image.
The adjustment made to each pixel's intensity is based on the cumulative distribution function (CDF) derived from the global histogram of all the pixel values in the image.
The histogram equalization is considered a pointwise operation because each pixel's new value is determined solely by its original value, without considering the values of neighboring pixels.
The mapping function, derived from the CDF, is applied directly to each pixel to determine its new intensity.
