# Faster R-CNN

**Faster R-CNN** further improved object detection by eliminating the need for external region proposals:

1. It introduced the **Region Proposal Network (RPN)**, a small CNN that generates region proposals directly from the feature map.
2. RPN and the main detection network share convolutional layers, leading to faster and more efficient training.
