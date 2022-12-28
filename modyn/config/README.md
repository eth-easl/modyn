# Configurations for the modyn package

## Modyn Configuration

TODO: Add a description of the configuration file

  trainer:
    type: object
    properties:
      epochs:
        type: string
        description: |
          Number of epochs to run the trainer for
      lr:
        type: string
        description: |
          The learning rate for the optimizer used for the training 
      batch_size:
        type: string
        description: |
          Number of elements to process in one batch
      train_set_size:
        type: string
        description: |
          The number of elements to be processed in one 'epoch'
      num_dataloader_workers:
        type: string
        description: |
          The number of data loaders to be used to fetch the training data
      model_config:
        type: object
        description: |
          Config used by the ML model
        properties:
          name:
            type: string
            description: |
              The model name to be used
          in_channels:
            type: string
            description: |
              Model specific property for the input number of channels
          num_classes:
            type: string
            description: |
              Model specific property for the number of output classes
          dropout:
            type: string
            description: |
              Model specific property for the dropout probability in the layers
          fc_in:
            type: string
            description: |
              Model specific property for the fc input
required:
  - project
  - input
  - storage
  - preprocess
  - odm
```
