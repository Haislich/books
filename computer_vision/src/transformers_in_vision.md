# Transformers in Vision

Transformers were originally developed for natural language processing (NLP) tasks but have since been adapted for vision tasks due to their ability to capture long-range dependencies and model complex relationships in data. While traditional models like CNNs and RNNs have limitations such as vanishing gradients or inability to process sequences in parallel, **Transformers** overcome these challenges using **self-attention** mechanisms.

## Transformers in Computer Vision

### How to Pass an Image to a Transformer?

1. **Treating Each Pixel as a Token**:
   - The basic idea is to treat every pixel in an image as a token (just like a word in an NLP task) and feed the image into a Transformer as a sequence of pixels.
   - **Problem**: This approach is computationally infeasible because self-attention, a core component of Transformers, has a complexity of **O(n²)**, where `n` is the number of tokens (i.e., pixels). For high-resolution images, this leads to extremely high computational costs.

2. **Using Downsampling**:
   - Downsampling the image before passing it into the Transformer reduces the sequence length, making computations more manageable.
   - **Drawback**: Downsampling can result in the loss of crucial details, especially fine-grained information, leading to less effective image representations.

3. **Dividing the Image into Patches**:
   - The standard approach today is to divide the image into smaller patches (e.g., 16x16 or 32x32 pixels). Each patch is treated as a token, and a **linear embedding layer** is applied to flatten the patch into a vector.
   - **Positional encoding** is then added to the embedding to maintain spatial information, as the Transformer is inherently position-agnostic.
   - This patch-based method significantly reduces computational costs while preserving important spatial information, making it the dominant method in **Vision Transformers (ViT)**.

## Transformers for Object Detection

One of the most innovative uses of Transformers in vision is for object detection, particularly in the **DETR (DEtection TRansformer)** model.

### DETR Network Components

1. **Convolutional Backbone**:
   - DETR begins with a traditional convolutional network (e.g., **ResNet-50** or **ResNet-101**) to extract features from the input image. The CNN reduces the image size and produces a set of feature maps that are passed to the Transformer.

2. **Transformer Encoder-Decoder**:
   - **Encoder**: The feature maps from the CNN are processed by a Transformer encoder, which applies **multi-head self-attention** to the patches. This allows the model to capture global information by relating every patch to every other patch, enabling it to model relationships between distant regions of the image.
   - **Decoder**: The decoder receives a fixed set of learnable **query embeddings**, each representing a potential object in the image. The decoder applies cross-attention between the query embeddings and the feature map, allowing it to predict objects' locations and classes by attending to relevant areas in the image.

3. **Classification and BBox Heads**:
   - For each query embedding, the decoder outputs a prediction, which is passed through separate **classification** and **bounding box regression heads**.
   - **Classification head**: Predicts the class label of the detected object.
   - **Bounding box head**: Predicts the coordinates of the object's bounding box.
   - Unlike traditional methods like **Faster R-CNN** or **YOLO**, which use anchor boxes and region proposals, DETR directly predicts bounding boxes without predefined anchors.

4. **Training with Hungarian Loss**:
   - DETR uses a unique training approach based on the **Hungarian algorithm** to match the predicted bounding boxes to the ground truth boxes. The matching is based on minimizing the **bipartite matching loss**, which ensures a one-to-one correspondence between predictions and actual objects in the image.

### Advantages of DETR

- **Unified Approach**: DETR integrates the object classification and bounding box prediction steps into a single model, unlike traditional methods that separate these tasks.
- **Global Context**: Thanks to the self-attention mechanism, DETR can model relationships between distant objects, something CNNs struggle to do.
- **Elimination of Anchors**: DETR removes the need for complex anchor box designs, simplifying the architecture and making training more straightforward.

### Disadvantages of DETR

- **Computational Efficiency**: The self-attention mechanism used in DETR has quadratic complexity, making it slower to train compared to traditional CNN-based object detectors.
- **Difficulty with Small Objects**: DETR sometimes struggles with detecting small or densely packed objects because the query embeddings may fail to represent them adequately.

## Transformers for Semantic Segmentation

**SegFormer** is a state-of-the-art architecture for semantic segmentation that leverages a Transformer-based backbone while maintaining efficiency and simplicity.

### SegFormer Architecture

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

### Advantages of SegFormer

- **Efficiency**: SegFormer achieves high performance while being computationally efficient, making it suitable for real-time applications and use on devices with limited resources (e.g., mobile devices).
- **Multi-scale Processing**: Its ability to handle information at different scales ensures good performance across various image types, whether detailed (urban scenes) or broad (satellite images).
- **Simplicity**: The lack of convolutions in the decoder simplifies implementation and reduces computational costs.

## Swin Transformers

**Swin Transformers** (Shifted Window Transformers) introduce a more efficient way of processing images using **hierarchical attention** and **local windowing**, making them highly scalable and suitable for high-resolution vision tasks like semantic segmentation, object detection, and classification.

### Architecture and Key Concepts of Swin Transformers

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

### Advantages of Swin Transformers

- **Computational Efficiency**: By using local windows and shifted windows, Swin Transformer reduces the computational burden, making it much more efficient than ViT for large images.
- **Scalability**: Its hierarchical structure makes it flexible for a variety of vision tasks, from low-resolution classification to high-resolution segmentation and detection.

## Transformers for Multi-Modal Learning

1. **CLIP (Contrastive Language-Image Pretraining)**:
   - **CLIP** is a multi-modal model that combines a **text encoder** and an **image encoder** to learn joint representations of text and images.
   - It is trained using a contrastive loss that maximizes the similarity between corresponding text-image pairs and minimizes the similarity between non-matching pairs.
   - **Applications**: CLIP can be used for zero-shot classification, where it classifies images based on text descriptions, even for classes it has not seen during training.

2. **MultiModal DETR**:
   - This model combines a **RoBERTa text encoder** (pretrained for language tasks) with a **CNN image encoder**. The outputs of both encoders are concatenated and fed into a Transformer that uses self-attention to model the relationship between the two modalities.
   - The model outputs prediction boxes for object detection tasks, combining both visual and textual information.
