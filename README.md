# SCHAF: Single-Cell Histology Analysis Framework

SCHAF is a framework for aligning single-cell RNA sequencing data with histology images, enabling spatial gene expression prediction from H&E images.

## Installation

1. Create and activate the conda environment:
```bash
conda env create -f schaf_environment.yml
conda activate final_schaf
```

## Usage

### 1. Running SCHAF Method (schaf_method.py)

The main SCHAF method supports both training and inference modes for various scenarios:

```bash
# Training mode example
python schaf_method.py --mode train \
    --scenario [mouse|cancer|htapp|placenta|lung] \
    --gpu 0 \
    --batch-size 32 \
    --workers 6 \
    --num-epochs 100 \
    --lr 0.001 \
    --tile-radius 112 \
    --use-wandb \
    --save-model

# Inference mode example
python schaf_method.py --mode inference \
    --scenario [mouse|cancer|htapp|placenta|lung] \
    --gpu 0 \
    --batch-size 32 \
    --workers 6
```

Key parameters:
- `--mode`: Choose between 'train' or 'inference'
- `--scenario`: Dataset scenario to use
- `--gpu`: GPU device ID to use
- `--batch-size`: Batch size for training/inference
- `--workers`: Number of data loading worker processes
- `--num-epochs`: Number of training epochs
- `--lr`: Learning rate
- `--tile-radius`: Radius of image tiles
- `--use-wandb`: Enable Weights & Biases logging
- `--save-model`: Save model checkpoints during training

### 2. Running Benchmarks (schaf_benchmarking.py)

The benchmarking module allows evaluation of SCHAF against other methods:

```bash
python schaf_benchmarking.py \
    --model-type [cycle|distance|caroline] \
    --dataset [mouse|cancer_in_sample|cancer_whole_sample] \
    --gpu 0 \
    --batch-size 32 \
    --epochs 100 \
    --save-dir ./benchmark_results
```

Key parameters:
- `--model-type`: Type of model to benchmark
- `--dataset`: Dataset to use for benchmarking
- `--gpu`: GPU device ID
- `--batch-size`: Batch size
- `--epochs`: Number of training epochs
- `--save-dir`: Directory to save benchmark results

### 3. Generating Figures (schaf_figures.ipynb)

The Jupyter notebook `schaf_figures.ipynb` contains code to generate benchmark comparison figures. To use:

1. Ensure you have inference results from both SCHAF and ground truth data
2. Open the notebook in JupyterLab:
```bash
jupyter lab
```

3. The notebook will generate four main benchmark comparison figures:
   - Cell type accuracy comparison (`supp2_bottom.pdf`)
   - Cell type accuracy ablation study (`celltype_ablation_paired.png`)
   - Full gene correlation distributions (`dists_benchmarks_paired.png`)
   - Ablation gene correlation distributions (`dists_ablation_paired.png`)

## Data Requirements

- For training: Paired H&E images and single-cell RNA sequencing data
- For inference: H&E images only
- For benchmarking: Ground truth data for comparison
- For figure generation: Inference results and ground truth data

## Citation

If you use SCHAF in your research, please cite our paper [paper citation to be added].

## License

[License information to be added] 
