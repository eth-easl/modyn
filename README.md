<div align="center">
<img src="docs/assets/logo.png" height=100 alt="Modyn logo"/>

---

[![GitHub Workflow Status](https://github.com/eth-easl/modyn/actions/workflows/workflow.yaml/badge.svg)](https://github.com/eth-easl/modyn/actions/workflows/workflow.yaml)
[![License](https://img.shields.io/github/license/eth-easl/modyn)](https://img.shields.io/github/license/eth-easl/modyn)

Modyn is an open-source platform for model training on dynamic datasets, i.e., datasets where points get added or removed over time.

</div>


## ⚡️ Quickstart

For deploying and running integration tests, you will need [Docker](https://docs.docker.com/get-docker/).
Furthermore, we use [mamba](https://mamba.readthedocs.io/en/latest/) for local environments and [tmuxp](https://github.com/tmux-python/tmuxp) for easily managing components panes (optional).
For local development, run
```bash
# In case you don't have mamba yet
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

# Start here if you have Mamba already
./initial_setup.sh
mamba env create -f ./environment.yml
mamba activate modyn
pip install -e .
pip install -r dev-requirements.txt
```
and then `./scripts/python_compliance.sh` to check that your local installation of Modyn is functioning.

If you want to run all Modyn components, run
```bash
./run_modyn.sh
tmuxp load tmuxp.yaml # Assumes you have tmuxp, see above for link
```

For running all integration tests, run
```bash
./run_integrationtests.sh
```

> **_macOS Installation:_**: Make sure to run `./initial_setup.sh` as outlined above. If not, installation might fail due to PyTorch not being found. Unfortunately, the PyTorch channel currently does not support macOS.

> **_GPU Installation:_**: If you want to use a GPU, make sure to install `nvidia-docker` and confirm to use CUDA on first run of `./initial_setup.sh`. Optionally, if you want to use Apex (require, e.g., for DLRM model), make sure to confirm to install Apex. In this case, having the NVIDIA docker runtime as Docker default runtime is required. The script will try to enable this, if we have sudo priviliges on the system. The CUDA version can be adjusted in the `initial_setup.sh` file.

**Next Steps**.
Checkout our [Example Pipeline](docs/EXAMPLE.md) guide for an example on how to run a Modyn pipeline.
Checkout our [Technical Guidelines](docs/TECHNICAL.md) for some hints on developing Modyn and how to add new data selection and triggering policies.
Checkout the [Architecture Documentation](docs/ARCHITECTURE.md) for an overview of Modyn's components.
Last, checkout our [vision paper on Modyn](https://anakli.inf.ethz.ch/papers/MLonDynamicData_EuroMLSys23.pdf) for an introduction to model training on dynamic datasets.

We are actively developing and designing Modyn, including more thorough documentation.
Please reach out via Github, Twitter, E-Mail, or any other channel of communication if you are interested in collaborating, have any questions, or have any problems running Modyn.

How to [contribute](docs/CONTRIBUTING.md).

## 🔁 What are dynamic datasets and what is Modyn used for?
ML is is often applied in use cases where training data evolves and/or grows over time, i.e., datasets are _dynamic_ instead of static.
Training must incorporate data changes for high model quality, however this is often challenging and expensive due to large datasets and models.
With Modyn, we are actively developing an open-source platform that manages dynamic datasets at scale and supports pluggable policies for when and what data to train on.
Furthermore, we are developing a representative open-source benchmarking suite for ML training on dynamic datasets.

The unit of execution in Modyn is a _pipeline_.
At minimum, a pipeline consists of (1) the model specification, (2) the training dataset, and a corresponding byte parsing function that defines how to convert raw sample bytes to model input, (3) the trigger policy, (4) the data selection policy, (5) training hyperparameters such as optimization criterion, optimizer, learning rate, batch size, and (6) training configuration such as data processing workers, whether to use automatic mixed precision, etc.
Checkout our [Example Pipeline](docs/EXAMPLE.md) guide for an example on how to run a Modyn pipeline.

Modyn allows researchers to explore training and data selection policies (see [Technical Guidelines](docs/TECHNICAL.md) on how to add new policies to Modyn), while alleviating the burdens of managing large dynamic datasets and orchestrating recurring training jobs.
However, we strive towards usage of Modyn in practical environments as well.
We welcome input from both research and practice.

## ✉️ About
Modyn is being developed at the [Efficient Architectures and Systems Lab (EASL)](https://anakli.inf.ethz.ch/#Group) at the [ETH Zurich Systems Group](https://systems.ethz.ch/).
Please reach out to `mboether [at] inf [­dot] ethz [dot] ch` or open an issue on Github if you have any questions or inquiry related to Modyn and its usage.
