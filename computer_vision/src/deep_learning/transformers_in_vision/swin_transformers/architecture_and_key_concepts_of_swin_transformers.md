# Architecture and Key Concepts of Swin Transformers

1. **Patch Partitioning**:
   - Similar to ViT, Swin Transformer starts by dividing the image into small patches. Each patch is linearly embedded into a fixed-dimension vector, representing a token that is fed into the Transformer.

2. **Moving Window (Shifted Window)**:
   - Instead of applying attention to the entire image, Swin Transformer calculates self-attention **within non-overlapping local windows**. This drastically reduces computational costs by limiting the attention mechanism to smaller regions.
   - To avoid the limitations of local windows, Swin introduces **shifted windows**, where the windows are shifted between layers to allow information to flow between adjacent windows. This ensures that the model captures long-range dependencies across the entire image.

3. **Hierarchical Architecture**:
   - Swin Transformer uses a **multi-scale** approach, building a pyramid-like representation of the image similar to CNNs (e.g., ResNet). Early layers focus on local details, while deeper layers capture more global context.
   - This hierarchical structure allows the model to efficiently scale to larger image sizes and tasks that require high-resolution inputs, such as segmentation.

4. **Concatenation and Upsampling**:
   - After processing the image at multiple scales, the outputs from different layers are **concatenated** and upsampled to produce a global representation. For tasks like semantic segmentation, the final output is returned to a higher resolution through upsampling layers.
