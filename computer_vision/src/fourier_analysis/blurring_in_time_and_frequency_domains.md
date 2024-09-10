# Blurring in Time and Frequency Domains

In the time domain, blurring is achieved by convolving an image with a blur kernel (a small matrix). The kernel is slid over the image, and at each position, the sum of the weighted pixel values covered by the kernel replaces the central pixel. The kernel's shape and size affect the blur characteristics. For example, a Gaussian kernel produces smooth gradients, while a square kernel tends to preserve edges, potentially introducing artifacts.

In the frequency domain, blurring is performed by multiplying the Fourier transform of the image by a filter (also in the frequency domain). A Gaussian filter in the frequency domain has a smooth decay, gradually attenuating high frequencies and preserving natural transitions. A square filter, however, has a sharp cutoff, eliminating high frequencies abruptly, which can introduce ringing artifacts (Gibbs phenomenon).
