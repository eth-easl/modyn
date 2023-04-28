## Steps to reproduce the bug


1. Download the data to the used directory. Install requires `gdown` and then run 
```python benchmark/yearbook/data_generation.py --dir /your/data/directory/yearbook```
2. Run modyn and start the trainer server 
3. Start the pipeline `benchmark/yearbook/runme.yaml` using ```modyn-supervisor --start-replay-at 0 benchmark/yearbook/runme.yaml modyn/config/examples/modyn_config.yaml```. It should fail around trigger 53.