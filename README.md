# SCHAF: Single-Cell Histology Analysis Framework

SCHAF is a framework for inferring single-cell resolution transcriptomic data from histology images, enabling gene expression prediction from H&E images.

## Installation

1. After having mamba and conda installed, create and activate the conda environment. This should take minutes.
```bash
mamba env create -f schaf_environment.yml
conda activate final_schaf
```

## Usage

### 1. Running SCHAF Method (schaf_method.py)

The main SCHAF method supports both training and inference modes for various scenarios:

#### Pre-defined Scenarios

```bash
# Training mode example for pre-defined datasets
python schaf_method.py --mode train \
    --scenario [mouse|cancer_in_sample|cancer_whole_sample|htapp|placenta|lung_cancer] \
    --fold [fold_id] \
    --gpu 0 \
    --batch-size 32 \
    --workers 6 \
    --num-epochs 100 \
    --lr 0.001 \
    --tile-radius 112 \
    --use-wandb \
    --save-model

# Inference mode example for pre-defined datasets
python schaf_method.py --mode inference \
    --scenario [mouse|cancer_in_sample|cancer_whole_sample|htapp|placenta|lung_cancer] \
    --fold [fold_id] \
    --gpu 0 \
    --batch-size 32 \
    --workers 6
```

#### Custom User Data

For your own H&E images and scRNA-seq/spatial transcriptomics data:

```bash
# Paired training (with spatial transcriptomics data)
python schaf_method.py --mode train \
    --scenario custom \
    --custom-data-dir /path/to/your/data \
    --custom-he-image your_he_image.tif \
    --custom-sc-file your_single_cell_data.h5ad \
    --custom-st-file your_spatial_data.h5ad \
    --custom-paired \
    --custom-scenario-name my_dataset \
    --fold 0 \
    --gpu 0 \
    --batch-size 32 \
    --num-epochs 100

# Unpaired training (scRNA-seq only)
python schaf_method.py --mode train \
    --scenario custom \
    --custom-data-dir /path/to/your/data \
    --custom-he-image your_he_image.tif \
    --custom-sc-file your_single_cell_data.h5ad \
    --custom-scenario-name my_dataset \
    --fold 0 \
    --gpu 0 \
    --batch-size 32 \
    --num-epochs 100

# Custom inference
python schaf_method.py --mode inference \
    --scenario custom \
    --custom-data-dir /path/to/your/data \
    --custom-he-image your_he_image.tif \
    --custom-scenario-name my_dataset \
    --fold 0 \
    --gpu 0
```

#### Key Parameters

- `--mode`: Choose between 'train' or 'inference'
- `--scenario`: Dataset scenario to use (or 'custom' for your own data)
- `--fold`: Fold/key to use as test set
- `--gpu`: GPU device ID to use
- `--batch-size`: Batch size for training/inference
- `--workers`: Number of data loading worker processes
- `--num-epochs`: Number of training epochs
- `--lr`: Learning rate
- `--tile-radius`: Radius of image tiles
- `--use-wandb`: Enable Weights & Biases logging
- `--save-model`: Save model checkpoints during training

#### Custom Data Parameters

- `--custom-data-dir`: Directory containing your data files
- `--custom-he-image`: Filename of your H&E image
- `--custom-sc-file`: Filename of your single-cell data (.h5ad format)
- `--custom-st-file`: Filename of your spatial transcriptomics data (.h5ad, for paired scenarios)
- `--custom-paired`: Use paired training (requires spatial transcriptomics data)
- `--custom-scenario-name`: Name for your custom scenario (used in output paths)

### 2. Running Benchmarks (schaf_benchmarking.py)

The benchmarking module allows evaluation of SCHAF against other methods:

```bash
python schaf_benchmarking.py \
    --model-type [cycle|distance|crossmodal] \
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

3. The notebook will generate the figures seen in the below-linked SCHAF study.

## Data Requirements

### Pre-defined Scenarios
- For training: Paired H&E images, single-cell RNA sequencing data, and, for Paired SCHAF, spatial transcriptomics data
- For inference: H&E images only
- For figure generation: Inference results and ground truth data

### Custom User Data

#### Required Files
- **H&E Image**: High-resolution histology image (`.tif`, `.png`, `.jpg` formats supported)
- **Single-cell Data**: AnnData object (`.h5ad`) containing single-cell RNA sequencing data
- **Spatial Transcriptomics Data** (for paired training): AnnData object (`.h5ad`) containing spatial transcriptomics data

#### Data Format Requirements

**Single-cell Data (`.h5ad`)**:
- `adata.X`: Gene expression matrix (cells × genes)
- `adata.var.index`: Gene names/symbols
- `adata.obs.index`: Cell barcodes/IDs
- Optional: `adata.obs['cluster']` for cell type annotations

**Spatial Transcriptomics Data (`.h5ad`)**:
- `adata.X`: Gene expression matrix (spots × genes)
- `adata.var.index`: Gene names/symbols (should overlap with single-cell data)
- `adata.obs.index`: Spot barcodes/IDs
- **Spatial coordinates** (required, one of):
  - `adata.obsm['spatial']`: 2D array with x,y coordinates
  - `adata.obs['x']` and `adata.obs['y']`: Separate columns for coordinates
- Optional: `adata.obs['cluster']` for spot annotations

#### Directory Structure
```
your_data_directory/
├── he_image.tif                    # H&E histology image
├── single_cell_data.h5ad          # Single-cell RNA-seq data
└── spatial_data.h5ad              # Spatial transcriptomics data (for paired)
```

#### Data Validation
Before training, SCHAF will automatically validate your data format:
```bash
# The validation will check:
# ✓ File existence
# ✓ Data format compatibility
# ✓ Gene overlap between datasets
# ✓ Spatial coordinate availability
# ✓ Data dimensions and quality
```

#### Preparing Your Data
1. **Convert to AnnData format**: Use `scanpy` to convert your data:
   ```python
   import scanpy as sc
   import pandas as pd
   
   # For single-cell data
   adata_sc = sc.read_csv("your_expression_matrix.csv").T
   adata_sc.var_names_unique()
   adata_sc.write("single_cell_data.h5ad")
   
   # For spatial data with coordinates
   adata_st = sc.read_csv("spatial_expression.csv").T
   coordinates = pd.read_csv("spatial_coordinates.csv")
   adata_st.obsm['spatial'] = coordinates[['x', 'y']].values
   adata_st.write("spatial_data.h5ad")
   ```

2. **Ensure gene name consistency**: Gene names should be consistent between datasets
3. **Check spatial coordinates**: Coordinates should be in pixel units matching the H&E image

## Citation

If you use SCHAF in your research, please cite our paper: https://www.biorxiv.org/content/10.1101/2023.03.21.533680v2.

## License

MIT License

## SCHAF-1M

SCHAF-1M dataset can be accessed at the following Zenodo link: https://zenodo.org/records/15611768?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA4Yjg3NjZkLWY0NWItNGE0OS1hYWIxLWQxZTkzYjYzOGMzMiIsImRhdGEiOnt9LCJyYW5kb20iOiI0NjU5MWVhMzc1MjU3YTNhNDU5MDIzMzU1ZjVlZjQyZiJ9.B14gC9v7iN8jG1-0vIbo_3Mc7LLKJ3IrwN8VH4-JE9XJDR4v8zpXleFU7E8YKKGCBMqKoY2EgB6O26wbA9RCdw
