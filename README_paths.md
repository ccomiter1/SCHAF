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

- **Mouse Xenium data**: Set `DATA_PATHS['mouse_xenium']` to your mouse data location
- **Cancer Xenium data**: Set `DATA_PATHS['cancer_xenium']` to your cancer data location
- **Single-cell datasets**: Update paths in `DATA_PATHS` for HTAPP, placenta, and lung cancer SC data
- **Histology images**: Update paths for histology image datasets

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