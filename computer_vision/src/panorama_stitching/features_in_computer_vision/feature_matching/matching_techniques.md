# Matching Techniques

- **Brute-Force Matching**:  
  The simplest method of feature matching, where each descriptor from one image is compared with every descriptor in the second image. While exhaustive and accurate, brute-force matching can be computationally expensive for large datasets.

- **Nearest Neighbor Matching**:  
  In this method, each descriptor from one image is matched with the descriptor from the second image that has the smallest distance. To improve robustness, techniques such as the ratio test (proposed by Lowe in SIFT) can be applied:
  - **Ratio Test**: Compares the distance of the nearest neighbor to the second nearest neighbor. If the ratio of the two distances is below a threshold, the match is accepted; otherwise, it is discarded to avoid false matches.

- **K-Nearest Neighbors (k-NN)**:  
  Instead of finding only the closest match, this method identifies the k closest matches for each feature. The ratio test can also be applied here to filter out ambiguous matches.

- **FLANN (Fast Library for Approximate Nearest Neighbors)**:  
  This is a more efficient alternative to brute-force matching, especially for large datasets. FLANN uses approximate nearest neighbor search algorithms, which trade off a slight loss in accuracy for a significant speed improvement.
