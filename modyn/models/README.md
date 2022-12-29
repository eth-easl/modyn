# Models

The 'BaseTrainer' is a base class for training functionality. It contains some basic functions such as saving and loading checkpoints, logging, etc. and defines the training loop.

### Custom models

The user defines custom models by extending the 'BaseTrainer' class. The custom class needs to set the model and optimizer objetcs, and populate the 'train_one_iteration' function. As an example, see the 'ResNet18' model.