# Fast R-CNN

**Fast R-CNN** improved on R-CNN by:

1. Applying a CNN to the entire image once.
2. Using **RoIPool** to extract fixed-size feature maps from the CNN output for each region proposal.
3. Using a lightweight MLP to classify and refine bounding boxes, significantly improving efficiency.
