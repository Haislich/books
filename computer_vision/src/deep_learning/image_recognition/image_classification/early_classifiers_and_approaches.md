# Early Classifiers and Approaches

1. **Nearest Neighbor**: This was one of the simplest classifiers. It involves finding the most similar image in a dataset (based on a distance metric like Euclidean distance) and assigning the same class label. While straightforward, it's computationally expensive and tends to perform poorly, especially as datasets grow larger and more diverse.

2. **Bag of Words (BoW)**:
   - BoW is an analogy to text processing, where visual words are created from image features.
   - **Feature extraction**: Using methods like SIFT (Scale-Invariant Feature Transform) or SURF, key points and descriptors are extracted from the image.
   - **K-means clustering** is applied to group these feature descriptors into a fixed number of clusters, creating a visual vocabulary.
   - Each image is then represented by a histogram of these visual word frequencies.
   - A classifier like SVM or Random Forest is trained on these histograms, making BoW one of the earliest and most successful methods for image classification before the deep learning era.
