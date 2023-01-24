# Metadata Processor

This is the Metadata Processor submodule.

This component receives metadata from the Metadata Collector running at the trainer server, processes it according to a chosen strategy, and saves it in the Metadata Database.

The metadata processor component is started using `modyn-metadata-processor config.yaml pipeline.yaml`.

The script should be in PATH after installing the `modyn` module.

## How to add a new strategy

Strating from the `abstract_processor_strategy` module, a new Strategy should extend the MetadataProcessorStrategy class and implement the `process_metadata` method. This method will receive a training ID and serialized JSON data. What is part of the data string depends on the Collector and the Selector's strategies. The method should return a dictionary:
```
{
	"keys": [ "string" ] | None,
	"seen": [ "boolean" ] | None,
	"data": [ "string" ] | None
}
```

