# Custom models

The user can define models here. The model definition should take as a parameter a 'model_configuration' dictionary with architecture-specific parameters. As an example, see the 'ResNet18' model.

# Wild Time models
The code for the models used for WildTime is taken from the official [repository](https://github.com/huaxiuyao/Wild-Time). 
The original version is linked in each class.
You can find [here](https://raw.githubusercontent.com/huaxiuyao/Wild-Time/main/LICENSE) a copy of the MIT license

# Embedding Recorder
Many coreset methods are adapted from the [DeepCore](https://github.com/PatrickZH/DeepCore/) library. To use them, the models must keep track of the embeddings (activations of the penultimate layer). This is
done using the `EmbeddingRecorder` class, which is adapted from the aforementioned project.
You can find a copy of their MIT license [here](https://raw.githubusercontent.com/PatrickZH/DeepCore/main/LICENSE.md)