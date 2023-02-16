# Metadata Processor

This is the Metadata Processor submodule.

This component receives metadata from the Metadata Collector running at the trainer server, processes it according to a chosen strategy, and saves it in the Metadata Database.

The metadata processor component is started using `modyn-metadata-processor config.yaml pipeline.yaml`.

The script should be in PATH after installing the `modyn` module.

## How to add a new strategy

Strating from the `abstract_processor_strategy` module, a new Strategy should extend the AbstractProcessorStrategy class and implement the two methods: `process_trigger_metadata` and `process_sample_metadata`.

Provided is a Basic Processor Strategy, which just takes the metadata from the Collector and persists it into the Metadata Database, without any changes.
