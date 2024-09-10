# Key Concepts in Contrastive Learning

1. **Positive Pairs**:
   - These are different views or augmentations of the same data point (e.g., two crops of the same image). The model should learn similar representations for these examples, as they represent the same content. Augmentations can include techniques like cropping, rotation, flipping, or color jittering.

2. **Negative Pairs**:
   - These are examples of different data points (e.g., two different images). The model should learn to distinguish between these, moving their representations apart in the feature space.

3. **Contrastive Loss**:
   - The model is trained using a **contrastive loss**, which minimizes the distance between the representations of positive pairs and maximizes the distance between negative pairs. A common variant of this loss is **NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy Loss), which efficiently separates positive and negative examples.
