# DETR Network Components

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
