# Local histogram equalization

Local histogram equalization is a technique used to enhance the contrast of images, similar to global histogram equalization, but with a key difference: it operates on small, localized regions of the image rather than on the entire image.

This method is particularly useful for improving the visibility of features in images that have varying lighting conditions across different areas.

In local histogram equalization, the resulting intensity values for a pixel are determined by the histogram of the region it belongs to, making it a neighborhood operation.
Each pixel's new intensity depends on the specific distribution of intensities within its local neighborhood or region, as defined by the small tiles or windows used in the process.
