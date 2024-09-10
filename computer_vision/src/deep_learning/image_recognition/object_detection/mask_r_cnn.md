# Mask R-CNN

**Mask R-CNN** extends Faster R-CNN by adding a third branch for pixel-level instance segmentation:

1. The RPN generates region proposals.
2. RoI Align ensures spatial accuracy for each region.
3. A segmentation branch produces binary masks for each detected object, making it suitable for tasks that require both detection and segmentation.

Mask R-CNN has become the standard for instance segmentation tasks due to its accuracy and flexibility.
