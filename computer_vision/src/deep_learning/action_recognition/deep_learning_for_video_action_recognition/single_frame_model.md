# Single Frame Model

In the single-frame approach:

- A series of **Convolutional Neural Networks (CNNs)** analyze individual frames of a video at different timesteps independently and in parallel.
- A **combination method** (often a small neural network) is used to combine the features from these CNNs, potentially using pooling layers to reduce the dimensionality of the feature maps and combine the outputs.

**Advantages**:

- Simpler to implement because each frame is treated as an independent image, avoiding complex temporal dependencies.
- Pooling layers help reduce the spatial dimensions of the feature maps, thus reducing computational cost.

**Disadvantages**:

- The model lacks **temporal awareness**, as pooling operates independently on each frame and doesn't capture the sequence's temporal order.
- Actions often depend on the sequence of frames, so ignoring temporal dependencies can lead to poor performance in recognizing dynamic actions.
