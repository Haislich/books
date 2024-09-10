# Fourier Analysis

Fourier analysis is a mathematical method used to decompose functions or signals into sinusoids of different frequencies.

Fourier analysis plays a crucial role in computer vision by providing methods to process, analyze, and manipulate images in the frequency domain. Some of the most important applications are:

- **Image Filtering**: Fourier analysis allows for efficient high-pass, low-pass, and band-pass filtering, which can be used to remove noise, blur, or enhance image features.
- **Image Compression**: By transforming an image into its frequency components, redundant or less important frequencies can be removed, leading to effective compression without significant loss of quality.
- **Feature Extraction**: Fourier transforms can highlight periodic patterns and textures in an image, which are useful for object recognition and classification.
- **Edge Detection**: Edges in images can be detected by focusing on specific frequency ranges within the Fourier transform of an image.

## Fourier Series

The Fourier series represents a periodic function as a sum of sine and cosine functions, or equivalently, as a sum of complex exponentials. A function \(f(t)\) that is periodic over a period \(T\) can be represented as:

\[
f(t) = a_0 + \sum_{n=1}^\infty \left(a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right)
\]

Where the Fourier coefficients \(a_n\) and \(b_n\) are given by:

\[
a_n = \frac{2}{T} \int_0^T f(t) \cos\left(\frac{2\pi n t}{T}\right) \, dt
\]
\[
b_n = \frac{2}{T} \int_0^T f(t) \sin\left(\frac{2\pi n t}{T}\right) \, dt
\]

## Fourier Transform

The Fourier transform extends the idea of Fourier series to non-periodic functions, providing a frequency spectrum for functions defined over all real numbers. The Fourier transform of a function \(f(t)\) is defined as:

\[
\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i \omega t} \, dt
\]

Where \(\omega\) is the angular frequency, and the integral is taken over the entire real line, capturing the function's values across time.

### Discrete Fourier Transform

In image processing, Fourier transforms are primarily used in their discrete form, known as the Discrete Fourier Transform (DFT). Since images do not naturally repeat across boundaries, they are treated as aperiodic signals, making the Fourier transform ideal for analyzing such non-periodic data.

The DFT converts an image to the frequency domain, facilitating operations such as filtering, enhancement, and noise reduction. For digital images, which consist of discrete pixel values, the DFT is used. The DFT is typically represented through its magnitude and phase components:

- **Magnitude Spectrum**: Shows the amplitude of each frequency component in the signal, indicating how much of each frequency is present in the original signal.
- **Phase Spectrum**: Indicates the shift or displacement of each frequency component relative to the start of the signal.

## Convolution Theorem

The Convolution Theorem states that convolution in the time domain corresponds to multiplication in the frequency domain, and vice versa. This is particularly useful in signal and image processing, as convolution can be computationally expensive. Using the Fourier transform, convolution can be performed more efficiently:

- Apply the Fourier transform to both the signal and the filter.
- Multiply their transforms in the frequency domain.
- Apply the inverse Fourier transform to the result.

This approach is much faster than directly performing convolution in the spatial domain, especially for large data sets.

## Blurring in Time and Frequency Domains

In the time domain, blurring is achieved by convolving an image with a blur kernel (a small matrix). The kernel is slid over the image, and at each position, the sum of the weighted pixel values covered by the kernel replaces the central pixel. The kernel's shape and size affect the blur characteristics. For example, a Gaussian kernel produces smooth gradients, while a square kernel tends to preserve edges, potentially introducing artifacts.

In the frequency domain, blurring is performed by multiplying the Fourier transform of the image by a filter (also in the frequency domain). A Gaussian filter in the frequency domain has a smooth decay, gradually attenuating high frequencies and preserving natural transitions. A square filter, however, has a sharp cutoff, eliminating high frequencies abruptly, which can introduce ringing artifacts (Gibbs phenomenon).

## Filters

- **Low Pass Filters (LPF)**: Allow low frequencies to pass through while attenuating high frequencies. In images, low frequencies represent smooth intensity variations, which are associated with general features and areas of uniform color. LPFs smooth the image, reducing noise and detail.
- **High Pass Filters (HPF)**: Allow high frequencies to pass while attenuating low frequencies. High frequencies represent rapid intensity changes, such as edges. HPFs enhance or detect edges, making them useful for edge detection and sharpening.
- **Band Pass Filters (BPF)**: Allow a specific band of frequencies to pass while blocking frequencies outside this range. BPFs can isolate frequencies within a chosen range, useful for focusing on specific details or textures in the image.

## Nyquist Sampling Theorem

The Nyquist Sampling Theorem states:

> To perfectly reconstruct a continuous signal from its samples, the sampling rate must be at least twice the highest frequency present in the signal (the Nyquist rate).

When a signal is undersampled (i.e., sampled below the Nyquist rate), higher frequencies are misinterpreted as lower frequencies, leading to distortions known as aliasing.

### Relation with Gaussian Pyramids

A Gaussian Pyramid is created by repeatedly reducing an image's resolution, applying Gaussian blurring at each step. This process produces a stack of progressively lower-resolution images. Gaussian blurring ensures that high-frequency content is removed before subsampling, which prevents aliasing and adheres to the Nyquist criterion.

### Relation to Human Vision

The human eye contains photoreceptor cells (rods and cones) that sample the continuous visual field. These cells are densely packed in the fovea (central region) and sparse toward the periphery, allowing for high-resolution sampling where focus is directed, akin to a higher sampling rate at the fovea. Just as in digital sampling, improper sampling in the visual system can lead to aliasing, such as when viewing fine patterns or distant scenes.

## Hybrid Images

Hybrid images combine two images using multi-resolution techniques. One image retains only high-frequency information (details), and the other retains only low-frequency information (general shape). These images are combined by simple addition:

- At close viewing distances, the human eye perceives the high-frequency details of the high-pass filtered image.
- At farther distances, the eye naturally filters out finer details, leaving the broader, low-frequency features from the low-pass filtered image.

This results in an image that changes interpretation based on the viewing distance.
