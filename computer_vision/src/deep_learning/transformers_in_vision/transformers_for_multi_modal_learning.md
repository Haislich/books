# Transformers for Multi-Modal Learning

1. **CLIP (Contrastive Language-Image Pretraining)**:
   - **CLIP** is a multi-modal model that combines a **text encoder** and an **image encoder** to learn joint representations of text and images.
   - It is trained using a contrastive loss that maximizes the similarity between corresponding text-image pairs and minimizes the similarity between non-matching pairs.
   - **Applications**: CLIP can be used for zero-shot classification, where it classifies images based on text descriptions, even for classes it has not seen during training.

2. **MultiModal DETR**:
   - This model combines a **RoBERTa text encoder** (pretrained for language tasks) with a **CNN image encoder**. The outputs of both encoders are concatenated and fed into a Transformer that uses self-attention to model the relationship between the two modalities.
   - The model outputs prediction boxes for object detection tasks, combining both visual and textual information.
