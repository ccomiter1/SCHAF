# SCHAF Path Configuration

This document explains how to configure SCHAF for your local environment by setting up the correct paths in `schaf_paths.py`.

## Quick Start

1. **Edit `schaf_paths.py`**: Update the base directory paths to match your system
2. **Validate paths**: Run the validation function to check your configuration
3. **Create directories**: Use the helper function to create necessary output directories

## Configuration Steps

### 1. Base Directory Setup

Edit the following variables in `schaf_paths.py`:

```python
# Main storage root - change this to match your storage setup
STORAGE_ROOT = "/your/storage/path"  # Change this!
STORAGE2_ROOT = "/your/storage2/path"  # Change this!
MOUNTS_ROOT = "/your/mounts/path"  # Change this!
MOUNTS_STORAGE2_ROOT = "/your/mounts_storage2/path"  # Change this!

# Project root directory
PROJECT_ROOT = f"{STORAGE_ROOT}/schaf_for_revision052424"
```

### 2. Data Directory Setup

Ensure your data is organized as expected. The key datasets needed are:

#### Pre-defined Scenarios
- **Mouse Xenium data**: Set `DATA_PATHS['mouse_xenium']` to your mouse data location
- **Cancer Xenium data**: Set `DATA_PATHS['cancer_xenium']` to your cancer data location
- **Single-cell datasets**: Update paths in `DATA_PATHS` for HTAPP, placenta, and lung cancer SC data
- **Histology images**: Update paths for histology image datasets

#### Custom User Data
For your own datasets, update the custom data paths in `CUSTOM_DATA_PATHS`:

```python
CUSTOM_DATA_PATHS = {
    'custom_he_images': "/your/path/to/he_images",
    'custom_sc_data': "/your/path/to/single_cell",
    'custom_st_data': "/your/path/to/spatial_transcriptomics",
    'custom_embeddings': "/your/path/to/embeddings",
    'custom_chunks': "/your/path/to/chunks",
    'custom_folds': "/your/path/to/folds",
    'custom_models': "/your/path/to/models",
    'custom_inferences': "/your/path/to/inferences",
}
```

### 3. Validation

Test your configuration:

```python
from schaf_paths import validate_paths, create_output_directories

# Check if paths exist and are accessible
results = validate_paths()
print(results)

# Create necessary output directories
create_output_directories()
```

### 4. Model and Output Directories

The following directories will be created automatically:

- **Model storage**: Where trained models are saved (`MODEL_PATHS`)
- **Output directories**: Where results are saved (`OUTPUT_PATHS`)
- **Embedding directories**: Where embeddings are stored (`EMBEDDING_PATHS`)
- **Temporary directories**: Working space for processing (`TEMP_PATHS`)

## Path Categories

### Base Paths (`BASE_PATHS`)
Root directories for different storage systems

### Model Paths (`MODEL_PATHS`)
- Pre-trained models (scGPT, ViT-UNI)
- Trained SCHAF models for different scenarios
- Model checkpoints and weights

### Data Paths (`DATA_PATHS`)
- Raw Xenium datasets (mouse, cancer)
- Single-cell datasets (HTAPP, placenta, lung cancer)
- Histology image datasets
- Metadata and clustering files

### Output Paths (`OUTPUT_PATHS`)
- Inference results
- Processed data outputs
- Cross-validation folds

### Embedding Paths (`EMBEDDING_PATHS`)
- Pre-computed histology embeddings
- Pre-computed single-cell embeddings

### Temporary Paths (`TEMP_PATHS`)
- Chunk processing directories
- Intermediate files during training

## Scenario-Specific Keys (`SCENARIO_KEYS`)
Sample identifiers and mappings for different experimental scenarios

## File Patterns (`FILE_PATTERNS`)
Naming conventions for models, data files, and outputs

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure you have read/write access to all directories
2. **Missing directories**: Run `create_output_directories()` to create them
3. **Wrong paths**: Use `validate_paths()` to check your configuration
4. **Data not found**: Verify your data is in the expected locations

### Path Detection

The `get_user_storage_root()` function attempts to auto-detect your storage setup based on the current working directory. If this fails, manually set `STORAGE_ROOT`.

### Validation Output

The validation function returns a dictionary with:
- `path`: The configured path
- `exists`: Whether the path exists
- `readable`: Whether you can read from the path
- `writable`: Whether you can write to the path

## Example Configuration

For a typical setup where your data is in `/data/username/`:

```python
STORAGE_ROOT = "/data/username"
STORAGE2_ROOT = "/data/username/storage2"
MOUNTS_ROOT = "/mnt/data/username"
MOUNTS_STORAGE2_ROOT = "/mnt/data/username/storage2"
PROJECT_ROOT = f"{STORAGE_ROOT}/schaf_for_revision052424"
```

## Getting Help

If you encounter issues:

1. Check the path validation results
2. Ensure all required data files are present
3. Verify directory permissions
4. Review the console output for specific error messages

Run `python schaf_paths.py` to see configuration instructions and current path status.

## Using Custom Data

SCHAF supports training on your own H&E images and scRNA-seq/spatial transcriptomics data through the custom scenario feature.

### Quick Start with Custom Data

1. **Organize your data**:
   ```
   /your/data/directory/
   ├── he_image.tif
   ├── single_cell_data.h5ad
   └── spatial_data.h5ad  # For paired training
   ```

2. **Update paths** (optional - can also specify via command line):
   ```python
   # In schaf_paths.py
   CUSTOM_DATA_PATHS['custom_st_data'] = "/your/data/directory"
   ```

3. **Run training**:
   ```bash
   python schaf_method.py --mode train \
       --scenario custom \
       --custom-data-dir /your/data/directory \
       --custom-he-image he_image.tif \
       --custom-sc-file single_cell_data.h5ad \
       --custom-st-file spatial_data.h5ad \
       --custom-paired \
       --fold 0 \
       --gpu 0
   ```

### Custom Scenario Configuration

You can also create reusable configurations for your datasets:

```python
from schaf_paths import create_custom_scenario_config

# Create configuration for your dataset
my_config = create_custom_scenario_config(
    scenario_name="my_tissue_dataset",
    data_dir="/path/to/my/data",
    he_image_filename="tissue_he.tif",
    is_paired=True,
    tile_radius=200
)
```

### Data Format Requirements

- **H&E Image**: Any standard image format (`.tif`, `.png`, `.jpg`)
- **Single-cell Data**: AnnData format (`.h5ad`) with:
  - `adata.X`: Expression matrix (cells × genes)
  - `adata.var.index`: Gene names
  - `adata.obs.index`: Cell IDs
- **Spatial Data**: AnnData format (`.h5ad`) with:
  - `adata.X`: Expression matrix (spots × genes)
  - `adata.obsm['spatial']` or `adata.obs['x']`/`adata.obs['y']`: Coordinates
  - Gene names matching single-cell data

### Validation

SCHAF automatically validates your custom data before training:

```python
from schaf_paths import validate_custom_data_format

results = validate_custom_data_format(
    "/path/to/data",
    "he_image.tif",
    "sc_data.h5ad",
    "st_data.h5ad"
)

for message in results['messages']:
    print(message)
``` 