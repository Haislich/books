# Fourier Analysis

Fourier analysis is a mathematical method used to decompose functions or signals into sinusoids of different frequencies.

Fourier analysis plays a crucial role in computer vision by providing methods to process, analyze, and manipulate images in the frequency domain.
Some of the most important applications are:

- Image Filtering: Fourier analysis allows for efficient high-pass, low-pass, and band-pass filtering, which can be used to remove noise, blur, or enhance image features.
- Image Compression: By transforming an image into its frequency components, redundant or less important frequencies can be removed, leading to effective compression without significant loss of quality.
- Feature Extraction: Fourier transforms can highlight periodic patterns and textures in an image, which are useful for object recognition and classification.
- Edge Detection: Edges in images can be detected by focusing on specific frequency ranges within the Fourier transform of an image.

## Fourier Series

The Fourier series is a way to represent a periodic function as a sum of sine and cosine functions, or equivalently, as a sum of complex exponentials.
A function $f(t)$ that is periodic over a period $T$ can be represented as:

$$
f(t) = a_0 + \sum_{n=1}^\infty \left(a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right) \\
a_n = \frac{2}{T} \int_0^T f(t) \cos\left(\frac{2\pi n t}{T}\right) dt\\
b_n = \frac{2}{T} \int_0^T f(t) \sin\left(\frac{2\pi n t}{T}\right) dt
$$

Where $a_n$ is the average value of the function over a period of time.

## Fourier Transform

The Fourier transform extends the idea of Fourier series to non-periodic functions, providing a frequency spectrum for functions defined over all real numbers.
The fourier transform of a function $f(t)$ can be defined as:

$$
\hat{f}(\omega) = \int_{-\infty}^\infty f(t) e^{-i \omega t} dt
$$

Where $\omega$ is the angluar frequency, and the integral is taken over the entire real time, reflectine the function's values across time.

### Discrete Fourier Transform

In image processing, Fourier Transforms are primarily used, specifically the Discrete Fourier Transform (DFT).

Images are typically treated as aperiodic signals because they don't naturally repeat across the boundaries.
The Fourier Transform is suited for analyzing such non-periodic data.
Many operations in image processing, such as filtering, enhancement, and noise reduction, are more intuitively and efficiently handled in the frequency domain.
The Fourier Transform provides the means to convert an image to this domain.
More in particular for digital images, which consist of discrete pixel values, the Discrete Fourier Transform (DFT) is used.

Visualizing the Discrete Fourier Transform (DFT) helps in understanding the frequency components of a signal or image.
The DFT is typically represented through its magnitude and phase components, which are extracted from the complex numbers resulting from the transformation.

- Magnitude Spectrum: The magnitude of the DFT provides information about the amplitude of each frequency component in the signal. It tells you how much of each frequency is present in the original signal.
- Phase Spectrum: The phase of the DFT indicates the shift or displacement of each frequency component relative to the start of the signal.

## Convolution theorem

The Convolution Theorem is a fundamental principle that connects the operations of convolution and multiplication through the Fourier transform, providing an efficient way to perform these operations in different domains.
Convolution can be computationally expensive, especially for large data sets.
The theorem allows convolution to be performed by multiplying the Fourier transforms of the functions and then applying the inverse Fourier transform to the result, which is typically more efficient.
Filters applied to signals or images often involve convolution with a kernel (or filter).
Using the Fourier transform, these convolutions can be executed more rapidly in the frequency domain.

The inverse relation also holds, which states that the inverse Fourier transform of the product of two Fourier transforms yields the convolution of their respective inverse Fourier transforms.
This property is useful for analyzing systems where you know the frequency response and need to determine the corresponding time-domain response.
<!-- For example, blurring an image can be accomplished by convolving the image with a Gaussian filter.
Instead of performing this convolution directly in the spatial domain, it is often more efficient to take the Fourier transform of both the image and the Gaussian filter, multiply these transforms, and then apply the inverse Fourier transform to the result. -->

## Blurring revised

In the time domain blurring is achieved by convolving an image with a blur kernel (a small matrix).
This kernel is slid over the image, and at each position, the sum of the weighted pixel values covered by the kernel replaces the central pixel.
The shape and size of the kernel affect the blur characteristics.
For example, a Gaussian kernel produces smooth gradients, while a square kernel (uniform filter) tends to preserve edges and corners more, which can result in artifacts.

In the frequency domain, blurring is achieved by multiplying the Fourier transform of the image by a filter (also in the frequency domain).
The nature of the filter determines how different frequencies are attenuated or enhanced.
A Gaussian filter in the frequency domain has a smooth decay, affecting high frequencies gradually and preserving natural transitions.
In contrast, a square filter has a sharp cutoff, abruptly eliminating high frequencies, which can introduce ringing artifacts (Gibbs phenomenon).

## Filters

- Low Pass Filters (LPF): Allow low frequencies to pass through while attenuating high frequencies. Low frequencies in images represent smooth variations in intensity, which are associated with general features and areas of uniform color. The image is smoothened, reducing noise and detail.
- High Pass Filtes (HPF): Allow high frequencies to pass while attenuating low frequencies. High frequencies represent rapid changes in intensity such as edges. Enhances or detects edges in the image, making it useful for edge detection and sharpening.
- Band Pass Filters (BPF): Allow a band of frequencies to pass through while blocking frequencies outside this range. This can be achieved by combining LPF and HPF. Isolates frequencies within a specific range, which can be used to focus on certain details or textures in the image.

## Nyquist sampling theorem

The Nyquist Sampling Theorem is a critical principle in signal processing that states:

> To perfectly reconstruct a continuous signal from its samples, the sampling rate must be at least twice the highest frequency present in the signal (the Nyquist rate).

When a signal is undersampled (sampled below the Nyquist rate), higher frequencies are incorrectly interpreted as lower frequencies, leading to distortions known as aliasing.

### Relation with gaussian pyramids

A Gaussian Pyramid is constructed by repeatedly reducing an image's resolution while applying a Gaussian blur between each step.
This process creates a stack of progressively lower resolution images.
When reducing the resolution of images in a pyramid, it's crucial to apply Gaussian blurring to respect the Nyquist criterion.
The blur removes high-frequency content from the image, which could otherwise cause aliasing when the image is subsampled.

### Relation to human vision

The human eye is filled with photoreceptor cells (rods and cones) that sample the continuous visual field.
These cells are not uniformly distributed across the retina but are denser in the fovea (central region) and sparser toward the periphery.
This distribution allows for high-resolution sampling of the visual field where focus is directed, akin to a higher sampling rate at the fovea conforming to the Nyquist rate for higher spatial frequencies (details).
Just as in digital sampling, improper sampling in the human visual system can lead to aliasing.
For instance, when looking at finely patterned materials or distant scenes without adequate focus, patterns can seem distorted or different from their true form.

#### Hybrid Images

Hybrid images combine two images into one, using multi-resolution techniques to present an image that changes interpretation based on viewing distance.
One image is processed to keep only high-frequency information (details).
Another image is processed to retain only low-frequency information (general shape and structure).
The high-pass filtered image and the low-pass filtered image are combined by simple addition.
The resulting image will have properties of both images—visible details from the high-pass image at close viewing distances, and broader features from the low-pass image at further distances.
At close range, the observer's vision is sensitive to the high-frequency details, thus perceiving the details of the high-pass image.
From a distance, the human eye naturally filters out the finer details (high frequencies), leaving the broader strokes (low frequencies) of the low-pass image to dominate the perception.
