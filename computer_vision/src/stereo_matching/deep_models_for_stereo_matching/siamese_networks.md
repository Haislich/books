# Siamese Networks

Siamese networks for stereo matching use convolutional neural networks (CNNs) to extract features from stereo image pairs. These features are then correlated, often via dot product, to measure similarity between corresponding points. Disparity is determined by selecting the maximum correlation score for each pixel, with options for further refinement using global optimization techniques. This deep learning approach improves robustness in handling occlusions and textureless regions compared to traditional methods.
