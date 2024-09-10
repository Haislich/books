# How to Pass an Image to a Transformer?

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
