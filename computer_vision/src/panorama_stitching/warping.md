# Warping

After computing the homography between two images, warping is the process of applying this transformation to align or stitch the images. Warping modifies the coordinates of pixels to fit a new coordinate system. There are two main types:

- **Forward Warping**: Directly applies the homography to each pixel in the source image, mapping it to a new position in the destination image. This can lead to issues like gaps (where no pixels are mapped) or overlaps (where multiple pixels map to the same destination). Post-processing is required to fill gaps or resolve overlaps, potentially affecting image quality.
  
- **Inverse Warping**: Iterates over each pixel in the destination image and uses the inverse of the homography to map back to the source image. This method avoids gaps by ensuring all destination pixels are accounted for and handles overlaps naturally.

Both methods involve interpolation since transformed coordinates often map to non-integer positions.
