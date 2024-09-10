# Limitations of FeedForward CNNs

When using traditional CNNs for action recognition, certain limitations arise:

1. **Static Analysis Window**: CNNs operate on a fixed number of frames (`L` frames), often treated as an independent sample.
2. **Sliding Window**: To analyze the entire video, a sliding window is used to move across the video by `t` steps at a time. However, each window is processed independently.

**Problems**:

- Increasing `L` to capture more frames results in an explosion in the number of parameters to learn, increasing the model's complexity and computational cost.
- CNNs are not inherently designed to process temporal sequences, so they are unaware of what happened in previous frames. This independence between timesteps makes it hard to capture long-range dependencies in actions.
- CNNs are poorly suited to handling variable-length sequences unless padding is used, which can be inefficient and introduce unnecessary complexity.
