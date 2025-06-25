#!/usr/bin/env python3
"""
SCHAF Paths Configuration

This file contains all the custom directory paths and file locations used by SCHAF.
Users should modify these paths to match their local environment and data organization.

The paths are organized into the following categories:
- Base directories: Root paths for different types of data
- Model directories: Locations where trained models are stored
- Data directories: Locations of input datasets
- Output directories: Where results and outputs are saved
- Embedding directories: Pre-computed or generated embeddings
- Temporary directories: Working directories for intermediate files

Usage:
    from schaf_paths import *
    
    # Or import specific sections:
    from schaf_paths import BASE_PATHS, MODEL_PATHS, DATA_PATHS
"""

import os

#######################################################################################
# Base Directory Configuration
# These are the root paths that other paths are built from
#######################################################################################

# Main storage root - change this to match your storage setup
STORAGE_ROOT = "/storage/ccomiter"
STORAGE2_ROOT = "/storage2/ccomiter"
MOUNTS_ROOT = "/mounts/stultzlab03/ccomiter"
MOUNTS_STORAGE2_ROOT = "/mounts/stultzlab03_storage2/ccomiter"

# Project root directory
PROJECT_ROOT = f"{STORAGE_ROOT}/schaf_for_revision052424"

# Base paths for different data types
BASE_PATHS = {
    'storage': STORAGE_ROOT,
    'storage2': STORAGE2_ROOT,
    'mounts': MOUNTS_ROOT,
    'mounts_storage2': MOUNTS_STORAGE2_ROOT,
    'project': PROJECT_ROOT,
}

#######################################################################################
# Model Storage Directories
# Where trained models and checkpoints are saved/loaded
#######################################################################################

MODEL_PATHS = {
    # Xenium/Cancer model directories
    'mouse_models': f"{PROJECT_ROOT}/data/xenium_cancer/mouse_models",
    'xenium_cancer_models': f"{PROJECT_ROOT}/data/xenium_cancer/xenium_cancer_models",
    
    # Unpaired scenario model directories
    'htapp_models': f"{PROJECT_ROOT}/data/xenium_cancer/htapp_models_with_annos",
    'placenta_models': f"{PROJECT_ROOT}/data/xenium_cancer/placenta_models_with_annos",
    'lung_cancer_models': f"{PROJECT_ROOT}/data/xenium_cancer/lung_cancer_models",
    
    # Pre-trained model components
    'scgpt_pancancer': f"{STORAGE_ROOT}/htapp_supervise/new_schaf_experiment_scripts/final_new_schaf_start_jan2324/scgpt_model/scGPT_pancancer/",
    'scgpt_human': f"{STORAGE_ROOT}/htapp_supervise/new_schaf_experiment_scripts/final_new_schaf_start_jan2324/scgpt_model/scGPT_human/",
    'vit_uni_checkpoint': f"{os.getcwd().split('/ccomiter/')[0]}/ccomiter/htapp_supervise/new_schaf_experiment_scripts/final_new_schaf_start_jan2324/VIT/UNI/checkpoint/",
}

#######################################################################################
# Input Data Directories
# Raw datasets and preprocessed data locations
#######################################################################################

DATA_PATHS = {
    # Xenium datasets
    'mouse_xenium': f"{STORAGE_ROOT}/all_xenium_new_data/mouse_pup_data",
    'cancer_xenium': f"{STORAGE_ROOT}/htapp_supervise/new_schaf_experiment_scripts/more_data/xenium",
    
    # Single-cell datasets
    'htapp_sc': f"{STORAGE_ROOT}/htapp_supervise/final_scs/schtapp",
    'placenta_sc': f"{STORAGE_ROOT}/htapp_supervise/final_scs/scplacenta",
    'lung_cancer_sc': f"{STORAGE2_ROOT}/metamia_data/rest_adatas/full_all_var_adatas",
    
    # Histology image datasets
    'htapp_hists': f"{MOUNTS_ROOT}/htapp_supervise/final_data0315/hists_may_good",
    'htapp_hist_info': "../../../htapp_supervise/final_data0315/all_hists_infos0315",
    'placenta_hists': "newest_bestest_placenta_hes",
    'lung_cancer_segmentation': f"{MOUNTS_ROOT}/schaf_for_revision052424/data/xenium_cancer/lung_cancer_segmentation",
    
    # Metadata and auxiliary files
    'placenta_metadata': f"{os.getcwd().split('ccomiter')[0]}ccomiter/schaf_for_revision052424/data/human_placenta/hplacenta_gene_matrix.h5ad",
    'cancer_clustering_base': f"{MOUNTS_ROOT}/schaf_for_revision052424/data/xenium_cancer/big_sample1_er_positive/analysis/clustering",
}

#######################################################################################
# Output Directories  
# Where results, inferences, and processed data are saved
#######################################################################################

OUTPUT_PATHS = {
    # Inference output directories
    'mouse_inferences': f"{PROJECT_ROOT}/data/xenium_cancer/mouse_inferences",
    'xenium_cancer_inferences': f"{PROJECT_ROOT}/data/xenium_cancer/xenium_cancer_inferences",
    'htapp_inferences': f"{PROJECT_ROOT}/data/xenium_cancer/htapp_inferences",
    'placenta_inferences': f"{PROJECT_ROOT}/data/xenium_cancer/placenta_inferences",
    'lung_cancer_inferences': f"{PROJECT_ROOT}/data/xenium_cancer/lung_cancer_inferences",
    
    # Processed data output directories
    'mouse_folds': f"{PROJECT_ROOT}/data/xenium_cancer/mouse_folds",
    'cancer_folds': f"{PROJECT_ROOT}/data/xenium_cancer/cancer_in_sample_folds",
}

#######################################################################################
# Embedding Directories
# Pre-computed or generated embedding storage locations
#######################################################################################

EMBEDDING_PATHS = {
    # Histology embeddings
    'htapp_hist_embeddings': "htapp_hist_embeddings",
    'placenta_hist_embeddings': "newest_bestest_placenta_hist_embeddings", 
    'lung_cancer_hist_embeddings': "lung_cancer_hist_embeddings",
    
    # Single-cell embeddings
    'htapp_sc_embeddings': "htapp_sc_embeddings",
    'placenta_sc_embeddings': "placenta_sc_embeddings",
    'lung_cancer_sc_embeddings': "lung_cancer_sc_embeddings",
}

#######################################################################################
# Temporary/Working Directories
# Intermediate processing and chunk storage
#######################################################################################

TEMP_PATHS = {
    # Chunk processing directories
    'mouse_chunks': f"{MOUNTS_STORAGE2_ROOT}/mouse_chunks",
    'cancer_chunks': f"{MOUNTS_STORAGE2_ROOT}/cancer_chunks",
    
    # Alternative chunk directories (if needed)
    'mouse_chunks_alt': f"{STORAGE2_ROOT}/mouse_chunks",
    'cancer_chunks_alt': f"{STORAGE2_ROOT}/cancer_chunks",
}

#######################################################################################
# Scenario-Specific Configuration Keys
# Keys and identifiers for different experimental scenarios
#######################################################################################

SCENARIO_KEYS = {
    'htapp': ['4531', '7179', '7479', '7629', '932', '6760', '7149', '4381', '8239'],
    'placenta': ['7', '8', '9', '11'],
    'placenta_codes': {'7': 'JS34', '8': 'JS40', '9': 'JS35', '11': 'JS36'},
}

#######################################################################################
# Custom User Data Configuration
# Template paths and settings for user-defined scenarios
#######################################################################################

# Custom data paths - users should modify these for their own datasets
CUSTOM_DATA_PATHS = {
    # Raw data directories
    'custom_he_images': f"{STORAGE_ROOT}/custom_data/he_images",
    'custom_sc_data': f"{STORAGE_ROOT}/custom_data/single_cell",
    'custom_st_data': f"{STORAGE_ROOT}/custom_data/spatial_transcriptomics",
    
    # Processed data directories
    'custom_embeddings': f"{STORAGE_ROOT}/custom_data/embeddings",
    'custom_chunks': f"{STORAGE_ROOT}/custom_data/chunks",
    'custom_folds': f"{STORAGE_ROOT}/custom_data/folds",
    
    # Output directories
    'custom_models': f"{PROJECT_ROOT}/data/custom/models",
    'custom_inferences': f"{PROJECT_ROOT}/data/custom/inferences",
}

# Template configuration for custom scenarios
CUSTOM_SCENARIO_TEMPLATE = {
    'data_dir': CUSTOM_DATA_PATHS['custom_st_data'],
    'he_path': 'he_image.tif',  # Filename of H&E image in data_dir
    'model_dir': CUSTOM_DATA_PATHS['custom_models'],
    'output_dir': CUSTOM_DATA_PATHS['custom_inferences'],
    'proj_dir': CUSTOM_DATA_PATHS['custom_folds'],
    'stage1_suffix': 'st.h5ad',
    'stage2_suffix': 'tang_proj.h5ad',
    'use_hold_out': True,
    'model_name_prefix': 'custom',
    'is_paired': True,  # Set to False for unpaired scenarios
    'tile_radius': 180,
    'needs_transform': False,  # Set to True if coordinate transformation needed
    'transform_file': None,  # Filename of transformation matrix if needed
}

#######################################################################################
# File Name Patterns and Extensions
# Common file naming conventions used throughout SCHAF
#######################################################################################

FILE_PATTERNS = {
    # Model file patterns
    'model_checkpoint': '{scenario}_{fold}_final_model.pth',
    'stage2_model': '{scenario}_stage2_leave_out_fold_{fold}_final_model.pth',
    'whole_sample_model': '{scenario}_stage2_whole_sample_final_model.pth',
    
    # Data file patterns
    'fold_st': 'fold_{fold}_st.h5ad',
    'fold_projection': 'fold_{fold}_tang_proj.h5ad',
    'chunk_projection': '{dataset}_{split_type}_chunk_{chunk_id}.h5ad',
    
    # Output file patterns
    'inference_result': 'fold_{fold}.h5ad',
    'whole_sample_result': 'whole_sample.h5ad',
}

#######################################################################################
# Helper Functions
# Utility functions for path manipulation and validation
#######################################################################################

def get_user_storage_root():
    """
    Get the user's storage root directory.
    
    This function attempts to detect the user's storage setup automatically.
    If automatic detection fails, it falls back to the configured STORAGE_ROOT.
    
    Returns:
        str: Path to the user's storage root directory
    """
    # Try to detect user from current working directory
    cwd = os.getcwd()
    if '/ccomiter/' in cwd:
        # Extract the base path before /ccomiter/
        base_path = cwd.split('/ccomiter/')[0]
        return f"{base_path}/ccomiter"
    else:
        # Fall back to configured path
        return STORAGE_ROOT

def validate_paths():
    """
    Validate that critical paths exist and are accessible.
    
    Returns:
        dict: Dictionary with path validation results
    """
    validation_results = {}
    
    # Check critical base directories
    critical_paths = [
        ('storage_root', STORAGE_ROOT),
        ('project_root', PROJECT_ROOT),
    ]
    
    for name, path in critical_paths:
        validation_results[name] = {
            'path': path,
            'exists': os.path.exists(path),
            'readable': os.access(path, os.R_OK) if os.path.exists(path) else False,
            'writable': os.access(path, os.W_OK) if os.path.exists(path) else False,
        }
    
    return validation_results

def create_output_directories():
    """
    Create all necessary output directories if they don't exist.
    
    This function should be called before running SCHAF to ensure
    all output directories are available.
    """
    directories_to_create = []
    
    # Add all output paths
    directories_to_create.extend(OUTPUT_PATHS.values())
    
    # Add all model paths
    directories_to_create.extend(MODEL_PATHS.values())
    
    # Add custom data paths
    directories_to_create.extend(CUSTOM_DATA_PATHS.values())
    
    # Add embedding directories (relative paths will be created in current directory)
    for embed_dir in EMBEDDING_PATHS.values():
        if not os.path.isabs(embed_dir):
            directories_to_create.append(embed_dir)
    
    # Create directories
    created_dirs = []
    failed_dirs = []
    
    for directory in directories_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
        except (OSError, PermissionError) as e:
            failed_dirs.append((directory, str(e)))
    
    return {
        'created': created_dirs,
        'failed': failed_dirs
    }

def create_custom_scenario_config(scenario_name: str, 
                                 data_dir: str,
                                 he_image_filename: str,
                                 is_paired: bool = True,
                                 tile_radius: int = 180,
                                 **kwargs) -> dict:
    """
    Create a custom scenario configuration for user data.
    
    Args:
        scenario_name (str): Name for the custom scenario
        data_dir (str): Path to the directory containing your data
        he_image_filename (str): Filename of the H&E image
        is_paired (bool): Whether this is a paired (with spatial transcriptomics) scenario
        tile_radius (int): Radius of image tiles to extract
        **kwargs: Additional configuration options
        
    Returns:
        dict: Configuration dictionary for the custom scenario
        
    Example:
        config = create_custom_scenario_config(
            scenario_name="my_dataset",
            data_dir="/path/to/my/data",
            he_image_filename="my_he_image.tif",
            is_paired=True,
            tile_radius=200
        )
    """
    config = CUSTOM_SCENARIO_TEMPLATE.copy()
    
    # Update with user-provided values
    config.update({
        'data_dir': data_dir,
        'he_path': he_image_filename,
        'model_dir': f"{PROJECT_ROOT}/data/{scenario_name}/models",
        'output_dir': f"{PROJECT_ROOT}/data/{scenario_name}/inferences",
        'proj_dir': f"{PROJECT_ROOT}/data/{scenario_name}/folds",
        'model_name_prefix': scenario_name,
        'is_paired': is_paired,
        'tile_radius': tile_radius,
    })
    
    # Update with any additional kwargs
    config.update(kwargs)
    
    return config

#######################################################################################
# User Configuration Instructions
#######################################################################################

def print_configuration_instructions():
    """
    Print instructions for users on how to configure paths for their environment.
    """
    instructions = """
    SCHAF Path Configuration Instructions
    ====================================
    
    To configure SCHAF for your environment, modify the following variables in schaf_paths.py:
    
    1. STORAGE_ROOT: Set this to your main storage directory
       Example: STORAGE_ROOT = "/your/storage/path"
    
    2. PROJECT_ROOT: Set this to where you want SCHAF data and results stored
       Example: PROJECT_ROOT = "/your/project/path/schaf_for_revision052424"
    
    3. Update other base paths (STORAGE2_ROOT, MOUNTS_ROOT, etc.) as needed
       for your storage configuration.
    
    4. Verify data paths in DATA_PATHS match your dataset locations:
       - mouse_xenium: Path to mouse Xenium data
       - cancer_xenium: Path to cancer Xenium data  
       - htapp_sc: Path to HTAPP single-cell data
       - placenta_sc: Path to placenta single-cell data
       - lung_cancer_sc: Path to lung cancer single-cell data
    
    5. Run validate_paths() to check if your configuration is correct:
       
       from schaf_paths import validate_paths
       results = validate_paths()
       print(results)
    
    6. Run create_output_directories() to create necessary output folders:
       
       from schaf_paths import create_output_directories
       create_output_directories()
    
    For more information, see the SCHAF documentation.
    """
    print(instructions)

if __name__ == "__main__":
    print_configuration_instructions()
    print("\nCurrent path validation results:")
    results = validate_paths()
    for name, info in results.items():
        status = "✓" if info['exists'] and info['readable'] else "✗"
        print(f"{status} {name}: {info['path']}") 