# LBBNN

Python package for **Bayesian neural networks with latent binary structure**, including both **LRT-based** and **FLOW-based** formulations with **input skip-connections**.

This repository provides implementations, utilities, and example scripts for experimenting with sparse Bayesian neural architectures designed to support both **predictive performance** and **structural interpretability**. In addition to the core models, the package includes tools for:

- synthetic data generation,
- training and evaluation,
- extraction of global network structure,
- local explanation,
- and visualization of learned sparse architectures.

It is also an R-package version of LBBNN, which can be found [here](https://github.com/LarsELund/LBBNN).

---

## Repository status

> **This repository is under active development.**

The codebase is already usable for experimentation and methodological work, but it should still be regarded as a **research software repository** rather than a finalized production package. Interfaces, module organization, and utility functions may still evolve as the implementation is refined and expanded.

Users should therefore expect that:

- some APIs may change,
- certain helper functions may be reorganized,
- additional examples and tests may be added,
- and documentation may be expanded as the project matures.

---

## Overview

The repository currently includes implementations of the following model families:

- **LRT-based LBBNNs**
  - Latent binary Bayesian neural networks (LBBNNs) using the local reparameterization trick (LRT),
  - with input skip-connections to support sparse but expressive architectures.

- **FLOW-based LBBNNs**
  - Latent binary Bayesian neural networks (LBBNNs) whose weight modeling incorporates normalizing-flow components,
  - again with input skip-connections, allowing more flexible posterior structure.

In addition, the package contains utilities for:

- generating synthetic data for experimentation,
- training and validating models,
- extracting and saving global structural summaries,
- generating local explanations for individual observations,
- and visualizing learned sparse networks.

---

## Installation

It is recommended to install the package in a dedicated virtual environment.

### Create and activate a virtual environment

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### Install in editable mode

From the repository root, install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plot]"
```

This is the recommended setup for development and experimentation, since changes made to the source code are immediately reflected without reinstalling the package.

---

## Running the test suite

To verify that the package is correctly installed and that the local environment is consistent with the current codebase, run:

```bash
python -m pytest -q
```

Running the test suite is strongly recommended after substantial modifications to the code.

---

## Basic usage

The package is designed to be used as a Python library. After installation, the main components can be imported directly from `LBBNN`.

### Example: LRT-based network

```python
import torch
from LBBNN import InputSkipLRTNetwork, create_data_unif

y, X = create_data_unif(n=64, classification=True, seed=1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = InputSkipLRTNetwork(
    dim=8,
    p=X.shape[1],
    hidden_layers=2,
    classification=True,
    n_classes=1,
    act_func=torch.relu,
)

with torch.no_grad():
    preds = model(X, ensemble=False)

print(preds.shape)
```

---

### Example: FLOW-based network

```python
import torch
from LBBNN import InputSkipFlowNetwork, create_data_unif

y, X = create_data_unif(n=64, classification=True, seed=1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = InputSkipFlowNetwork(
    dim=8,
    p=X.shape[1],
    hidden_layers=2,
    num_transforms=2,
    classification=True,
    n_classes=1,
    act_func=torch.relu,
)

with torch.no_grad():
    preds = model(X, ensemble=False)

print(preds.shape)
print(model.kl())
```

---

## Global structure and local explanation

One of the main motivations for the repository is not only to train sparse Bayesian networks, but also to inspect and interpret their learned structure.

The package includes functionality for:

- extracting global sparse structure,
- computing structural summary metrics,
- saving active network connectivity,
- producing graph-based structure visualizations,
- and generating local explanations for individual observations.

Typical imports include:

```python
from LBBNN import plotting
from LBBNN import local_explain_piecewise_linear_act
```

Example usage:

```python
metrics = plotting.get_metrics(model)
print(metrics)

x_explain = X[0]

explanation, preds, p = local_explain_piecewise_linear_act(
    net=model,
    input_data=x_explain,
    median=True,
    sample=True,
    n_samples=32,
    magnitude=True,
    include_potential_contribution=False,
    n_classes=1,
)
```

---

## Example scripts

The repository contains example scripts in the `examples/` directory. These scripts are intended to provide minimal, reproducible demonstrations of how the package can be used.

Typical examples include:

- basic usage of the **LRT-based** network,
- basic usage of the **FLOW-based** network,
- and usage of the plotting / explanation utilities.

Example commands:

```bash
python examples/basic_usage.py
python examples/flow_usage.py
python examples/plotting_demo.py
python examples/run_flow_experiment.py
python examples/run_lrt_experiment.py
```

These scripts may serve as a useful starting point for custom experiments, benchmark studies, or exploratory analyses.

---

## Repository structure

A representative repository structure is:

```text
LBBNN/
├── README.md
├── pyproject.toml
├── LICENSE
├── examples/
│   ├── basic_usage.py
│   ├── flow_usage.py
│   └── plotting_demo.py
│   └── run_flow_experiment.py
│   └── run_lrt_experiment.py
├── tests/
│   ├── test_data.py
│   ├── test_model_forward.py
│   ├── test_training_utils.py
│   ├── test_flow_forward.py
│   ├── test_flow_kl_and_compat.py
│   └── test_plotting.py
└── LBBNN/
        ├── __init__.py
        ├── data.py
        ├── inspection.py
        ├── explain.py
        ├── training.py
        ├── lrt/
        │   ├── __init__.py
        │   ├── layers.py
        │   ├── network.py
        ├── flow/
        │   ├── __init__.py
        │   ├── transforms.py
        │   ├── layers.py
        │   └── network.py
        └── plotting/
            ├── __init__.py
            ├── _common.py
            ├── graphs.py
            ├── explanations.py
            ├── images.py
            └── metrics.py
```

---

## Main modules

### Core model implementations

- `LBBNN.lrt.network`  
  LRT-based latent binary Bayesian neural network with input skip-connections.

- `LBBNN.lrt.layers`  
  LRT-based Bayesian linear layer definitions.

- `LBBNN.flow.network`  
  FLOW-based latent binary Bayesian neural network with input skip-connections.

- `LBBNN.flow.layers`  
  FLOW-based Bayesian linear layers.

- `LBBNN.flow.transforms`  
  Normalizing-flow components used in the FLOW-based network.

---

### Supporting modules

- `LBBNN.data`  
  Synthetic dataset generation for experimental studies.

- `LBBNN.training`  
  Training, validation, and evaluation helpers.

- `LBBNN.inspection`  
  Utilities for structural inspection and extraction of sparse connectivity.

- `LBBNN.explain`  
  Local explanation methods.

- `LBBNN.plotting`  
  Plotting and visualization helpers for structure and explanations.

---

## Intended use

This repository is intended primarily for:

- methodological experimentation,
- reproducible research,
- analysis of sparse Bayesian neural network architectures,
- and development of interpretable Bayesian models with learned latent structure.

It may also be useful as a starting point for downstream applications where a sparse, interpretable neural network architecture is desirable.

---

## Development workflow

A recommended development workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```

This setup supports iterative development, testing, and experimentation.

---

## Notes on reproducibility

As with most research software, results may depend on:

- random seeds,
- hardware configuration,
- PyTorch / dependency versions,
- and specific experiment settings.

Where possible, users are encouraged to:

- set seeds explicitly,
- save model checkpoints and outputs,
- and use the provided example and experiment scripts as templates for reproducible workflows.

---

## License

This project is distributed under the **MIT License**.

---

## Citation

If you use this repository in academic research, please cite the relevant methodological paper as well as the software repository or archived software release.

The LBBNN Python package is developed based on a handful of papers.

[Sparse Bayesian Neural Networks: Bridging Model and Parameter Uncertainty through Scalable Variational Inference](https://www.mdpi.com/2227-7390/12/6/788):

```bibtex
@article{hubin2024sparse,
  title={Sparse Bayesian neural networks: bridging model and parameter uncertainty through scalable variational inference},
  author={Hubin, Aliaksandr and Storvik, Geir},
  journal={Mathematics},
  volume={12},
  number={6},
  pages={788},
  year={2024},
  publisher={MDPI}
}
```

[Sparsifying Bayesian neural networks with latent binary vari
ables and normalizing flows](https://openreview.net/pdf?id=d6kqUKzG3V):

```bibtex
@article{skaaret-lund2024sparsifying,
  title={Sparsifying Bayesian neural networks with latent binary variables and normalizing flows},
  author={Lars Skaaret-Lund and Geir Storvik and Aliaksandr Hubin},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
}
```

[Explainable Bayesian deep learning through input-skip
Latent Binary Bayesian Neural Networks](https://arxiv.org/pdf/2503.10496) (preprint):

```bibtex
@article{hoyheim2025explainable,
  title={Explainable Bayesian deep learning through input-skip Latent Binary Bayesian Neural Networks},
  author={H{\o}yheim, Eirik and Skaaret-Lund, Lars and S{\ae}b{\o}, Solve and Hubin, Aliaksandr},
  journal={arXiv preprint arXiv:2503.10496},
  year={2025}
}
```

### Software citation

```bibtex
@software{hoyheim_lbbnn,
  author       = {Eirik H{\o}yheim},
  title        = {LBBNN: Latent Binary Bayesian Neural Networks with input-skip},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/eirihoyh/LBBNN}
}
```