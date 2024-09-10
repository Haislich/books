# Region-Based CNN (R-CNN)

**R-CNN** broke ground by combining region proposals with CNN feature extraction:

1. An external algorithm (e.g., selective search) generates region proposals.
2. Each region is cropped and resized to a fixed size.
3. A CNN is applied to extract features, which are then classified with a linear SVM and refined using bounding box regression.

R-CNN provided state-of-the-art results but was computationally heavy due to repeated CNN computations for each region.
