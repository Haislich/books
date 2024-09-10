# DispNet

DispNet is the first end-to-end deep neural network designed for stereo matching. Its architecture, inspired by U-Net, includes a contracting path to capture context and an expanding path for precise localization, with skip connections for better detail retention. A correlation layer processes disparities to compute similarity between patches of stereo images. Using multi-scale loss, DispNet is trained to handle varying levels of difficulty, improving accuracy across diverse stereo vision tasks.
