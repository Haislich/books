# Introduction of RNNs

To handle the sequential nature of video data, **Recurrent Neural Networks (RNNs)** are introduced. RNNs are capable of processing sequential data and maintaining memory of past inputs by weighing their computations on previous timesteps.

- In this approach, **CNNs** are first used to extract features from each frame, and then these features are passed to an **RNN** (e.g., LSTM or GRU), which processes the sequence of frame features over time.

**Advantages**:

- RNNs naturally handle variable-length sequences and can retain information over time, allowing the model to learn temporal dependencies.
- RNNs excel at capturing patterns across sequential data, making them suitable for tasks where temporal dynamics are critical (e.g., action recognition).

**Disadvantages**:

- RNNs are not **parallelizable**, meaning they must process sequences step-by-step. This results in slower training times compared to CNNs, which can process multiple frames in parallel.
- **Long-range dependencies** can be challenging to learn for vanilla RNNs, although LSTMs and GRUs help mitigate this issue to some extent.
