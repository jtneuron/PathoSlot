# PathoSlot

Official implementation of PathoSlot, accepted by MICCAI 2026.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n pathoslot python=3.10
conda activate pathoslot

pip install torch numpy pandas h5py pyyaml tqdm matplotlib seaborn scikit-learn scikit-survival
```

Install the CUDA-enabled PyTorch build that matches your system if you plan to train on GPU.

## Data Format

The training scripts expect:

- A split CSV/TSV with sample IDs, fold columns, biomarker labels, and survival labels.
- A task config YAML. See `my_datatsets/zhongshan_config.yaml` for the expected fields.
- One `.h5` embedding file per sample ID in the embedding directory.

Each `.h5` file should contain:

- `vis_features`
- `text_features`
- `name_embedding`

## Run

Set paths with environment variables:

```bash
export PATHOSLOT_SPLIT=/path/to/split.csv
export PATHOSLOT_TASK_CONFIG=/path/to/config.yaml
export PATHOSLOT_EMBEDDINGS_DIR=/path/to/embeddings
export PATHOSLOT_RESULTS_DIR=/path/to/output_dir
```

Train and evaluate:

```bash
python train_slot.py
```

Evaluate an existing checkpoint:

```bash
python test_slot.py
```

By default, the scripts fall back to repository-relative placeholder paths under `data/` and `outputs/`.
