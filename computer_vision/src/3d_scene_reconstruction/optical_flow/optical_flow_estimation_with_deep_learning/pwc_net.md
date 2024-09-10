# PWC-Net

PWC-Net (Pyramid, Warping, Cost volume) builds on several innovations for optical flow estimation. It processes images through a pyramid of feature representations, capturing motion at multiple scales. Large displacements are handled at coarser levels, while finer details are refined at higher resolutions.

The **warping** step aligns the second image with the first using the flow estimated from coarser levels, simplifying displacement estimation. A **cost volume** is then computed to measure the similarity between the warped image and the first image, helping estimate motion.

Starting from the coarsest level, PWC-Net iteratively refines the optical flow at each level of the pyramid, with the final output being the full-resolution flow field.
