# Performance Evaluation: Intersection Over Union (IoU)

IoU is the primary metric for evaluating object detection performance:

1. **Intersection**: The overlapping area between the predicted bounding box and the ground truth box.
2. **Union**: The combined area covered by both boxes.
3. **IoU**: Calculated as the ratio of the intersection over the union.

- **IoU = 1**: Perfect detection.
- **IoU = 0**: No overlap, completely incorrect detection.

IoU is often used as a threshold (e.g., IoU > 0.5) to determine whether a detection is considered a true positive.
