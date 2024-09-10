# Subsequent Developments

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
