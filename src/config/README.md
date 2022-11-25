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
  input:
    type: object
    properties:
      adapter:
        type: string
        description: |
          The input adapter to use for the tests.
      send_batch_size:
        type: integer
        description: |
          The number of samples in a batch to send to the system.
      send_batch_interval:
        type: integer
        description: |
          The interval in seconds between sending batches to the system.
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
  preprocess:
    type: object
    properties:
      port:
        type: string
        description: |
          The port to use for the preprocess server.
      hostname:
        type: string
        description: |
          The hostname to use for the preprocess server.
      function:
        type: string
        description: |
          The function to use for the preprocess server. # TODO - change this
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
required:
  - project
  - input
  - storage
  - newqueue
  - preprocess
  - odm
```
