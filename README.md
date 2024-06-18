# Implemented-transformer

"I had a great time implementing the Transformer model and learned a lot in the process. The Transformer is an encoder-decoder model that involves several key modules, which I carefully studied and implemented. These modules include:


* **The Embedding Layer**: The goal of an embedding layer is to enable a model to learn more about the relationships between words, tokens, or other inputs. This embedding layer can be viewed as transforming data from a higher-dimension space to a lower-dimension space.

* **Positional Encoding**: adds information about the position of each token in the input sequence, which is important for the model to understand the order of the tokens.

* **Multi-Head Attention**: allows the model to focus on different parts of the input sequence simultaneously, improving its ability to capture complex dependencies between tokens.

* **Position-Wise Feed-Forward Network**: a fully connected neural network that is applied to each position in the input sequence independently, allowing for non-linear transformations of the input.

* **Layer Normalization**: helps to stabilize the training of the model by normalizing the activations of each layer.

* **The Encoder**: a stack of identical layers that process the input sequence and extract contextual features.

* **The Decoder**: a stack of identical layers that use the contextual features extracted by the encoder to generate the output sequence.

and finally, built the complete Transformer model.

This repo uses the transformer to translate between languages, but it is also used for other natural language processing tasks and computer vision. This implementation will cover the **translation task**.
