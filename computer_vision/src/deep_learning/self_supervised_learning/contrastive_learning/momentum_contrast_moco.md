# Momentum Contrast (MoCo)

**Momentum Contrast (MoCo)** is a contrastive learning framework designed to address the challenge of maintaining a large number of negative examples. MoCo introduces two key innovations:

1. **Dynamic Queue**:
   - MoCo maintains a queue of negative examples that is dynamically updated during training. This queue stores past examples and avoids the need to recompute negatives in each training batch, ensuring a diverse set of negatives over time.

2. **Momentum Encoder**:
   - Instead of updating the model’s representations in every training step, MoCo uses a **momentum encoder**, which updates its representations slowly based on past versions of the model. This stabilizes the representations of negative examples, improving training efficiency.
