# SuperPoint and SuperGlue

**SuperPoint** is a convolutional neural network (CNN) that simultaneously detects interest points and computes descriptors. It is pre-trained on synthetic data and fine-tuned on real-world images using self-supervised learning.

- **Dual Head Architecture**: A shared CNN encoder extracts feature maps for both interest point detection and descriptor computation. One head produces a heatmap for keypoint detection, and the other outputs a dense descriptor map for describing keypoints.
- **Keypoint Detection**: Local maxima in the heatmap identify keypoints, with non-maximum suppression ensuring they are well-distributed.
- **Descriptor Generation**: Descriptors, extracted from the corresponding keypoints in the descriptor map, are designed to be distinctive and robust against image transformations.

**SuperGlue** is a feature matching method that enhances SuperPoint's output using a Graph Neural Network (GNN) to improve matching accuracy, especially in challenging conditions (e.g., large viewpoint changes or occlusions).

- **Graph Construction**: Descriptors from SuperPoint (or another detector) are treated as graph nodes, with edges representing potential matches between two images.
- **Graph Neural Network**: The GNN refines matches by considering both local descriptor similarities and the global geometric consistency of features across the graph.
- **Match Filtering**: SuperGlue selects matches that are both locally and globally consistent, outperforming traditional nearest neighbor methods in complex scenarios.
