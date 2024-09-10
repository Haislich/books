# SegFormer Architecture

1. **Compact Transformer Backbone**:
   - SegFormer uses a novel **Mix Transformer (MiT)** backbone, which is designed to be more efficient than traditional Transformer architectures like ViT or DETR.
   - **Hierarchical multi-scale processing**: MiT processes input images at multiple scales, allowing it to capture both local details and global context. This helps the model perform well on both fine-grained and large-scale structures.

2. **No Convolutions in the Decoder**:
   - Unlike traditional segmentation models that use convolutions in the decoder, SegFormer’s decoder is fully Transformer-based, making it more lightweight.
   - The decoder takes the multi-scale features from the backbone and fuses them into a single representation, which is used to predict the segmentation mask.

3. **Multi-Scale Feature Fusion**:
   - A core strength of SegFormer is its ability to **fuse multi-scale features** effectively. By doing so, it captures fine details at lower levels and the broader context at higher levels, leading to more accurate segmentation results.

4. **Direct Prediction of Masks**:
   - The decoder directly predicts the class of each pixel in the image, outputting a **segmentation map** that assigns a label to every pixel.
