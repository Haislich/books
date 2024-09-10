# Solution: Learnable Upsampling

1. **Low-resolution output**: Downsampling layers (e.g., max-pooling) reduce the spatial resolution of feature maps, making it difficult to produce fine-grained segmentation outputs.
2. **Small receptive field**: Convolutions, when applied repeatedly, reduce the network’s ability to capture large context (global information) about the image.
