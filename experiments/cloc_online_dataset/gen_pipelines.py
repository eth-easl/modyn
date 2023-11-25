import pathlib

PIPELINE_BLANK = """
pipeline:
    name: cloc_{0}_{1}_{2}_{3}
    description: CLOC Training.
    version: 1.0.0
model:
  id: ResNet50
  config:
    num_classes: 713
model_storage:
  full_model_strategy:
    name: "PyTorchFullModel"
training:
  gpus: 1
  device: "cuda:0"
  amp: False
  dataloader_workers: {0}
  num_prefetched_partitions: {1}
  parallel_prefetch_requests: {2}
  use_previous_model: True
  initial_model: random
  initial_pass:
    activated: False
  batch_size: 256
  optimizers:
    - name: "default"
      algorithm: "SGD"
      source: "PyTorch"
      param_groups:
        - module: "model"
          config:
            lr: 0.025
            weight_decay: 0.0001
            momentum: 0.9
  optimization_criterion:
    name: "CrossEntropyLoss"
  checkpointing:
    activated: False
  selection_strategy:
    name: NewDataStrategy
    maximum_keys_in_memory: {3}
    config:
      storage_backend: "database"
      limit: -1
      reset_after_trigger: True
data:
  dataset_id: cloc
  transformations: ["transforms.RandomResizedCrop(224)",
                    "transforms.RandomHorizontalFlip()",
                    "transforms.ToTensor()",
                    "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"]
  bytes_parser_function: |
    from PIL import Image
    import io
    def bytes_parser_function(data: bytes) -> Image:
      return Image.open(io.BytesIO(data)).convert("RGB")
trigger:
  id: DataAmountTrigger
  trigger_config:
    data_points_for_trigger: 5000000
"""

def main():
    curr_dir = pathlib.Path(__file__).resolve().parent
    for num_dataloader_workers in [16,1,2,8]:
        for partition_size in [5000, 25000, 100000, 1000000]:
            for num_prefetched_partitions in [0,1,2,6]:
                for parallel_pref in [1,2,4,8]:
                    if num_prefetched_partitions == 0 and parallel_pref > 1:
                        continue

                    if num_prefetched_partitions > 0 and parallel_pref > num_prefetched_partitions:
                        continue
                    
                    if partition_size == 5000:
                        if num_dataloader_workers not in [1,16]:
                            continue
                        
                        if num_prefetched_partitions in [2]:
                            continue
                    
                    pipeline = PIPELINE_BLANK.format(num_dataloader_workers, num_prefetched_partitions, parallel_pref, partition_size)
                    
                    with open(f"{curr_dir}/pipelines/cloc_{num_dataloader_workers}_{num_prefetched_partitions}_{parallel_pref}_{partition_size}.yml", "w") as pfile:
                        pfile.write(pipeline)


if __name__ == "__main__":
    main()