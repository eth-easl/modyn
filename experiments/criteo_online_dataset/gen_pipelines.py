import pathlib

PIPELINE_BLANK = """
pipeline:
    name: criteo_{0}_{1}_{2}_{3}
    description: DLRM/Criteo Training.
    version: 1.0.0
model:
  id: DLRM
  config: # these parameters are consistent with the parameters used for the experiments shown in the NVIDIA repo:  https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
    embedding_dim: 128
    interaction_op: "cuda_dot"
    hash_indices: False
    bottom_mlp_sizes: [512, 256, 128]
    top_mlp_sizes: [1024, 1024, 512, 256, 1]
    embedding_type: "joint_fused"
    num_numerical_features: 13
    use_cpp_mlp: True
    categorical_features_info:
      cat_0: 7912889
      cat_1: 33823
      cat_2: 17139
      cat_3: 7339
      cat_4: 20046
      cat_5: 4
      cat_6: 7105
      cat_7: 1382
      cat_8: 63
      cat_9: 5554114
      cat_10: 582469
      cat_11: 245828
      cat_12: 11
      cat_13: 2209
      cat_14: 10667
      cat_15: 104
      cat_16: 4
      cat_17: 968
      cat_18: 15
      cat_19: 8165896
      cat_20: 2675940
      cat_21: 7156453
      cat_22: 302516
      cat_23: 12022
      cat_24: 97
      cat_25: 35
model_storage:
  full_model_strategy:
    name: "PyTorchFullModel"
training:
  gpus: 1
  device: "cuda:0"
  amp: True
  dataloader_workers: {0}
  num_prefetched_partitions: {1}
  parallel_prefetch_requests: {2}
  use_previous_model: True
  initial_model: random
  initial_pass:
    activated: False
  batch_size: 65536
  optimizers:
    - name: "mlp"
      algorithm: "FusedSGD"
      source: "APEX"
      param_groups:
        - module: "model.top_model"
          config:
            lr: 24
        - module: "model.bottom_model.mlp"
          config:
            lr: 24
    - name: "opt_1"
      algorithm: "SGD"
      source: "PyTorch"
      param_groups:
        - module: "model.bottom_model.embeddings"
          config:
            lr: 24
  lr_scheduler:
    name: "DLRMScheduler"
    source: "Custom"
    optimizers: ["mlp", "opt_1"]
    config:
      base_lrs: [[24, 24], [24]]
      warmup_steps: 8000
      warmup_factor: 0
      decay_steps: 24000
      decay_start_step: 48000
      decay_power: 2
      end_lr_factor: 0
  optimization_criterion:
    name: "BCEWithLogitsLoss"
  grad_scaler_config:
    growth_interval: 1000000000
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
  dataset_id: criteo_tiny
  bytes_parser_function: |
    import torch
    import numpy as np
    def bytes_parser_function(x: bytes) -> dict:
      num_features = x[:52]
      cat_features = x[52:]
      num_features_array = np.frombuffer(num_features, dtype=np.float32)
      cat_features_array = np.frombuffer(cat_features, dtype=np.int32)
      return {{
        \"numerical_input\": torch.asarray(num_features_array, copy=True, dtype=torch.float32),
        \"categorical_input\": torch.asarray(cat_features_array, copy=True, dtype=torch.long)
      }}
  label_transformer_function: |
    import torch
    # we need to convert our integer-type labels to floats,
    # since the BCEWithLogitsLoss function does not work with integers.
    def label_transformer_function(x: torch.Tensor) -> torch.Tensor:
      return x.to(torch.float32)
trigger:
  id: DataAmountTrigger
  trigger_config:
    data_points_for_trigger: 30000000
"""

def main():
    curr_dir = pathlib.Path(__file__).resolve().parent
    for num_dataloader_workers in [16,1,2,8]:
        for partition_size in [10000, 100000, 2500000, 5000000]:
            for num_prefetched_partitions in [0,1,2,6]:
                for parallel_pref in [1,2,4,8]:
                    if num_prefetched_partitions == 0 and parallel_pref > 1:
                        continue

                    if num_prefetched_partitions > 0 and parallel_pref > num_prefetched_partitions:
                        continue
                    
                    if partition_size == 10000:
                        if num_dataloader_workers not in [1,16]:
                            continue
                        
                        if num_prefetched_partitions in [2]:
                            continue
                    
                    pipeline = PIPELINE_BLANK.format(num_dataloader_workers, num_prefetched_partitions, parallel_pref, partition_size)
                    
                    with open(f"{curr_dir}/pipelines_new/criteo_{num_dataloader_workers}_{num_prefetched_partitions}_{parallel_pref}_{partition_size}.yml", "w") as pfile:
                        pfile.write(pipeline)


if __name__ == "__main__":
    main()