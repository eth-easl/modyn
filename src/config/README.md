# Dynamic Datasets DSL Project Configuration

The configuration file is located in `src/config/config.yaml`. The file is structured as follows:

```yaml
%YAML 1.1
---
$schema: "http://json-schema.org/draft-04/schema"
id: "http://stsci.edu/schemas/yaml-schema/draft-01"
title:
  Dynamic Datasets DSL Project Configuration
description: |
  This is the configuration file for the Dynamic Datasets DSL Project.
  It contains the configuration for the system, adapt as required.
properties:
  project:
    type: object
    properties:
      name:
        type: string
        description: |
          The name of the project.
      description:
        type: string
        description: |
          The description of the project.
      version:
        type: string
        description: |
          The version of the project.
  storage:
    type: object
    properties:
      adapter:
        type: string
        description: |
          The storage adapter to use for the tests.
      port:
        type: string
        description: |
          The port to use for the storage server.
      hostname:
        type: string
        description: |
          The hostname to use for the storage server.
      data_source:
        type: object
        description: |
          The data source to use for the storage server.
        properties:
          enabled:
            type: boolean
            description: |
              Whether the data source is enabled.
          type:
            type: string
            description: |
              The name of the data source.
            optional: true
          batch_size:
            type: integer
            description: |
              The batch size to use for the data source.
            optional: true
          batch_interval:
            type: integer
            description: |
              The batch interval to use for the data source.
            optional: true
      dbm:
        type: object
        properties:
          path:
            type: string
            description: |
              The path to the DBM file.
      s3:
        type: object
        properties:
          bucket:
            type: string
            description: |
              The name of the S3 bucket.
          access_key:
            type: string
            description: |
              The access key to use for the S3 bucket.
          secret_key:
            type: string
            description: |
              The secret key to use for the S3 bucket.
          endpoint_url:
            type: string
            description: |
              The endpoint URL to use for the S3 bucket.
      sqlite:
        type: object
        properties:
          path:
            type: string
            description: |
              The path to the SQLite file.
      postgresql:
        type: object
        properties:
          host:
            type: string
            description: |
              The hostname to use for the PostgreSQL server.
          port:
            type: string
            description: |
              The port to use for the PostgreSQL server
          database:
            type: string
            description: |
              The name of the PostgreSQL database.
          user:
            type: string
            description: |
              The user to use for the PostgreSQL server.
          password:
            type: string
            description: |
              The password to use for the PostgreSQL server.
  newqueue:
    type: object
    properties:
      port:
        type: string
        description: |
          The port to use for the newqueue server.
      hostname:
        type: string
        description: |
          The hostname to use for the newqueue server.
      polling_interval:
        type: integer
        description: |
          The polling interval to use for the newqueue server.
      postgresql:
        type: object
        properties:
          host:
            type: string
            description: |
              The hostname to use for the PostgreSQL server.
          port:
            type: string
            description: |
              The port to use for the PostgreSQL server
          database:
            type: string
            description: |
              The name of the PostgreSQL database.
          user:
            type: string
            description: |
              The user to use for the PostgreSQL server.
          password:
            type: string
            description: |
              The password to use for the PostgreSQL server.
  odm:
    type: object
    properties:
      port:
        type: string
        description: |
          The port to use for the odm server.
      hostname:
        type: string
        description: |
          The hostname to use for the odm server.
      postgresql:
        type: object
        properties:
          host:
            type: string
            description: |
              The hostname to use for the PostgreSQL server.
          port:
            type: string
            description: |
              The port to use for the PostgreSQL server
          database:
            type: string
            description: |
              The name of the PostgreSQL database.
          user:
            type: string
            description: |
              The user to use for the PostgreSQL server.
          password:
            type: string
            description: |
              The password to use for the PostgreSQL server.
  ptmp:
    type: object
    properties:
      port:
        type: string
        description: |
          The port to use for the ptmp server.
      hostname:
        type: string
        description: |
          The hostname to use for the ptmp server.
      processor:
        type: string
        description: |
          The processor to use for the ptmp server.

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
  - newqueue
  - preprocess
  - odm
```
