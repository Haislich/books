# Action Recognition

Action recognition in videos aims to classify actions, gestures, or interactions by analyzing temporal sequences of frames. This is a challenging task because it requires models to account for both the spatial (individual frame) and temporal (sequence of frames) dimensions.

## Deep Learning for Video Action Recognition

Action recognition in videos can be approached using various deep learning models, which typically fall under the umbrella of **activity classification**. The goal is to identify a predefined set of actions in videos, such as human gestures, sports activities, or interactions, from sequences of frames.

### Single Frame Model

In the single-frame approach:

- A series of **Convolutional Neural Networks (CNNs)** analyze individual frames of a video at different timesteps independently and in parallel.
- A **combination method** (often a small neural network) is used to combine the features from these CNNs, potentially using pooling layers to reduce the dimensionality of the feature maps and combine the outputs.

**Advantages**:

- Simpler to implement because each frame is treated as an independent image, avoiding complex temporal dependencies.
- Pooling layers help reduce the spatial dimensions of the feature maps, thus reducing computational cost.

**Disadvantages**:

- The model lacks **temporal awareness**, as pooling operates independently on each frame and doesn't capture the sequence's temporal order.
- Actions often depend on the sequence of frames, so ignoring temporal dependencies can lead to poor performance in recognizing dynamic actions.

### Multiple Frames

To capture temporal information, various strategies are introduced that incorporate multiple frames into the analysis:

- **SingleFrame with Temporal Dimension Information Fusion**: Incorporates time by fusing features extracted from multiple frames over time. This method retains the simplicity of analyzing individual frames while incorporating some time context.
- **Late Fusion**: Extracts features from individual frames and combines them at a later stage (e.g., before classification).
- **Slow Fusion**: Instead of simply combining the frames, slow fusion incrementally incorporates information from frames over time, allowing the model to gradually build temporal awareness across frames.

However, the precise implementation of these fusion strategies varies, and further elaboration on how the temporal information is fused is not always provided in the literature.

### Limitations of FeedForward CNNs

When using traditional CNNs for action recognition, certain limitations arise:

1. **Static Analysis Window**: CNNs operate on a fixed number of frames (`L` frames), often treated as an independent sample.
2. **Sliding Window**: To analyze the entire video, a sliding window is used to move across the video by `t` steps at a time. However, each window is processed independently.

**Problems**:

- Increasing `L` to capture more frames results in an explosion in the number of parameters to learn, increasing the model's complexity and computational cost.
- CNNs are not inherently designed to process temporal sequences, so they are unaware of what happened in previous frames. This independence between timesteps makes it hard to capture long-range dependencies in actions.
- CNNs are poorly suited to handling variable-length sequences unless padding is used, which can be inefficient and introduce unnecessary complexity.

### Introduction of RNNs

To handle the sequential nature of video data, **Recurrent Neural Networks (RNNs)** are introduced. RNNs are capable of processing sequential data and maintaining memory of past inputs by weighing their computations on previous timesteps.

- In this approach, **CNNs** are first used to extract features from each frame, and then these features are passed to an **RNN** (e.g., LSTM or GRU), which processes the sequence of frame features over time.

**Advantages**:

- RNNs naturally handle variable-length sequences and can retain information over time, allowing the model to learn temporal dependencies.
- RNNs excel at capturing patterns across sequential data, making them suitable for tasks where temporal dynamics are critical (e.g., action recognition).

**Disadvantages**:

- RNNs are not **parallelizable**, meaning they must process sequences step-by-step. This results in slower training times compared to CNNs, which can process multiple frames in parallel.
- **Long-range dependencies** can be challenging to learn for vanilla RNNs, although LSTMs and GRUs help mitigate this issue to some extent.

### Subsequent Developments

1. **2D CNN + RNN**: Instead of processing a single frame at a time, sequences of frames are passed as input to 2D CNNs. The CNNs extract spatial features from each frame, and the RNN captures the temporal relationships between them.

2. **3D CNNs**:
   - 3D Convolutional Neural Networks add an extra temporal dimension to the filters, making them capable of directly capturing spatio-temporal information.
   - Instead of treating each frame as an independent image, 3D CNNs process video clips, typically consisting of **16 frames**. This number of frames is sufficient to provide a receptive field that captures the temporal dynamics.
   - 3D convolutional filters operate across the video clip’s dimensions, processing height, width, and time simultaneously (e.g., a filter of size `KxMxNxD` for a video of shape `TxHxWxD`).
   - The output of each clip is a high-dimensional feature vector (typically 4096 dimensions), which serves as a video descriptor.

3. **Two-Stream Networks**:
   - This approach splits the video into two streams: a **spatial stream** (which analyzes individual frames) and a **temporal stream** (which analyzes the motion in the video, often using optical flow).
   - Each stream processes the frames or motion information independently, and the outputs are fused at a later stage.
   - This method effectively captures both appearance information (from the spatial stream) and motion dynamics (from the temporal stream).

4. **Inflated 3D CNN (I3D)**:
   - I3D extends traditional 2D CNNs, which are typically trained on static image datasets like ImageNet, by inflating the 2D convolutional filters into 3D filters.
   - This "inflation" is achieved by repeating the pre-trained 2D filters along a third (temporal) dimension, allowing the network to leverage the vast amount of knowledge gained from large image datasets and apply it to video-based tasks.
   - I3D uses an **Inception** architecture, which enhances efficiency by processing video data through a deep and wide network, capturing both fine details and larger-scale patterns across time.

### Soft vs Hard Attention

Attention mechanisms allow a model to focus on specific parts of the input (in this case, frames or regions of a video) when making predictions. These mechanisms are crucial in action recognition, where different frames or parts of a video may have varying importance.

1. **Hard Attention**:
   - Hard attention focuses on a single region of the input (e.g., a specific region of a frame or a single frame in a sequence).
   - It does not use gradient-based learning, so it is often trained using **reinforcement learning** methods, which makes training more complex and less efficient.
   - Hard attention is computationally cheaper but may miss important contextual information by focusing too narrowly.

2. **Soft Attention**:
   - Soft attention calculates a **weighted combination** of all inputs (e.g., all frames or regions of a frame), allowing the model to attend to multiple parts of the input simultaneously.
   - Soft attention is differentiable, meaning it can be trained using backpropagation, making it more suitable for most deep learning architectures.
   - This method improves performance in tasks that require focus on multiple regions or frames, such as recognizing complex actions spread across a sequence of frames.

3. **Self-Attention** (Intra-Attention):
   - Self-attention mechanisms analyze different positions in a sequence to compute the relationships between them, generating a unitary representation of the sequence.
   - Self-attention is highly efficient because it allows **parallel updates** to the input embeddings, unlike RNNs, which require sequential updates.
   - This method is widely used in models like **Transformers**, which have recently shown state-of-the-art performance in video action recognition tasks due to their ability to model long-range dependencies efficiently.
