# SimCLR (Simple Framework for Contrastive Learning of Visual Representations)

**SimCLR** is another popular contrastive learning framework developed by Google Research. It simplifies the contrastive learning process while achieving state-of-the-art performance.

1. **Data Augmentation**:
   - SimCLR relies on extensive **data augmentation**, generating positive pairs by applying different transformations to the same image. This could involve cropping, color jittering, or flipping.

2. **Projection Network**:
   - After extracting features from the image using a CNN backbone, SimCLR applies a **projection head** that maps the learned features into an embedding space. This projection network helps improve the quality of the learned representations by focusing on relevant features for contrastive learning.

3. **Large Batch Sizes**:
   - A limitation of SimCLR is that it requires very large batch sizes to provide a sufficient number of negative examples in each batch. This can be computationally expensive, requiring large-scale hardware.
