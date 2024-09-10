# Sliding Window Detection

This method involves:

1. Sliding a fixed-size window across the image at different scales and positions.
2. Extracting features from each window (e.g., HOG features).
3. Classifying the content of each window using a classifier like SVM.
4. Non-maximum suppression to eliminate overlapping detections.

Sliding windows were the foundation of traditional object detection but are computationally expensive and inefficient.
