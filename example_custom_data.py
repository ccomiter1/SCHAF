#!/usr/bin/env python3
"""
Example: Using SCHAF with Custom Data

This script demonstrates how to prepare and use your own H&E images and 
scRNA-seq/spatial transcriptomics data with SCHAF.

Before running this example:
1. Install SCHAF dependencies
2. Configure paths in schaf_paths.py
3. Prepare your data in the expected format
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from schaf_paths import validate_custom_data_format, create_custom_scenario_config

def prepare_example_data(output_dir: str):
    """
    Create example data files for demonstration.
    Replace this with your actual data preparation code.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating example data in {output_dir}")
    
    # Create example single-cell data
    n_cells = 1000
    n_genes = 2000
    
    # Generate random expression data
    expression_matrix = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create AnnData object
    adata_sc = sc.AnnData(X=expression_matrix)
    adata_sc.obs.index = [f"cell_{i}" for i in range(n_cells)]
    adata_sc.var.index = [f"gene_{i}" for i in range(n_genes)]
    
    # Add some metadata
    adata_sc.obs['cluster'] = np.random.choice(['cluster_A', 'cluster_B', 'cluster_C'], n_cells)
    
    # Save single-cell data
    sc_path = os.path.join(output_dir, "single_cell_data.h5ad")
    adata_sc.write(sc_path)
    print(f"✓ Created single-cell data: {sc_path}")
    
    # Create example spatial transcriptomics data
    n_spots = 500
    
    # Generate spatial coordinates
    x_coords = np.random.uniform(0, 1000, n_spots)
    y_coords = np.random.uniform(0, 1000, n_spots)
    
    # Generate expression data (subset of genes from single-cell)
    spatial_genes = adata_sc.var.index[:1500]  # Use subset of genes
    spatial_expression = np.random.negative_binomial(3, 0.4, size=(n_spots, len(spatial_genes)))
    
    # Create spatial AnnData object
    adata_st = sc.AnnData(X=spatial_expression)
    adata_st.obs.index = [f"spot_{i}" for i in range(n_spots)]
    adata_st.var.index = spatial_genes
    
    # Add spatial coordinates
    adata_st.obsm['spatial'] = np.column_stack([x_coords, y_coords])
    adata_st.obs['cluster'] = np.random.choice(['region_1', 'region_2', 'region_3'], n_spots)
    
    # Save spatial data
    st_path = os.path.join(output_dir, "spatial_data.h5ad")
    adata_st.write(st_path)
    print(f"✓ Created spatial data: {st_path}")
    
    # Create a dummy H&E image (replace with your actual image)
    he_image = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    
    # Save as image file (you would use your actual H&E image here)
    from PIL import Image
    he_path = os.path.join(output_dir, "he_image.tif")
    Image.fromarray(he_image).save(he_path)
    print(f"✓ Created H&E image: {he_path}")
    
    return sc_path, st_path, he_path

def validate_data_example(data_dir: str):
    """
    Example of how to validate your custom data before training.
    """
    print("\n" + "="*50)
    print("VALIDATING CUSTOM DATA")
    print("="*50)
    
    # Validate the data format
    validation_results = validate_custom_data_format(
        data_dir=data_dir,
        he_image="he_image.tif",
        sc_file="single_cell_data.h5ad",
        st_file="spatial_data.h5ad"
    )
    
    # Print validation results
    print("Validation Results:")
    for message in validation_results['messages']:
        print(f"  {message}")
    
    if validation_results['valid']:
        print("\n✅ Data validation passed! Your data is ready for SCHAF training.")
    else:
        print("\n❌ Data validation failed. Please check the error messages above.")
        return False
    
    # Print data information
    if 'data_info' in validation_results:
        info = validation_results['data_info']
        print(f"\nData Summary:")
        if 'sc_cells' in info:
            print(f"  Single-cell: {info['sc_cells']} cells, {info['sc_genes']} genes")
        if 'st_spots' in info:
            print(f"  Spatial: {info['st_spots']} spots, {info['st_genes']} genes")
    
    return True

def create_config_example(data_dir: str):
    """
    Example of how to create a custom scenario configuration.
    """
    print("\n" + "="*50)
    print("CREATING CUSTOM CONFIGURATION")
    print("="*50)
    
    # Create custom scenario configuration
    config = create_custom_scenario_config(
        scenario_name="my_tissue_example",
        data_dir=data_dir,
        he_image_filename="he_image.tif",
        is_paired=True,  # We have spatial transcriptomics data
        tile_radius=180
    )
    
    print("Custom scenario configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def training_command_examples(data_dir: str):
    """
    Print example commands for training with custom data.
    """
    print("\n" + "="*50)
    print("TRAINING COMMAND EXAMPLES")
    print("="*50)
    
    print("1. Paired training (with spatial transcriptomics):")
    print(f"""
python schaf_method.py --mode train \\
    --scenario custom \\
    --custom-data-dir {data_dir} \\
    --custom-he-image he_image.tif \\
    --custom-sc-file single_cell_data.h5ad \\
    --custom-st-file spatial_data.h5ad \\
    --custom-paired \\
    --custom-scenario-name my_tissue_example \\
    --fold 0 \\
    --gpu 0 \\
    --batch-size 32 \\
    --num-epochs 50 \\
    --lr 0.001 \\
    --save-model
""")
    
    print("2. Unpaired training (single-cell only):")
    print(f"""
python schaf_method.py --mode train \\
    --scenario custom \\
    --custom-data-dir {data_dir} \\
    --custom-he-image he_image.tif \\
    --custom-sc-file single_cell_data.h5ad \\
    --custom-scenario-name my_tissue_example \\
    --fold 0 \\
    --gpu 0 \\
    --batch-size 32 \\
    --num-epochs 50
""")
    
    print("3. Inference:")
    print(f"""
python schaf_method.py --mode inference \\
    --scenario custom \\
    --custom-data-dir {data_dir} \\
    --custom-he-image he_image.tif \\
    --custom-scenario-name my_tissue_example \\
    --fold 0 \\
    --gpu 0
""")

def main():
    """
    Main example workflow.
    """
    print("SCHAF Custom Data Example")
    print("=" * 50)
    
    # Set up example data directory
    example_data_dir = "./example_custom_data"
    
    print(f"This example will create sample data in: {example_data_dir}")
    print("In practice, you would replace this with your actual data directory.\n")
    
    # Step 1: Prepare example data (replace with your data preparation)
    print("Step 1: Preparing example data...")
    sc_path, st_path, he_path = prepare_example_data(example_data_dir)
    
    # Step 2: Validate the data
    print("\nStep 2: Validating data format...")
    if not validate_data_example(example_data_dir):
        return
    
    # Step 3: Create configuration
    print("\nStep 3: Creating custom configuration...")
    config = create_config_example(example_data_dir)
    
    # Step 4: Show training commands
    print("\nStep 4: Example training commands...")
    training_command_examples(example_data_dir)
    
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("1. Replace the example data with your actual H&E images and scRNA-seq data")
    print("2. Update the paths in schaf_paths.py if needed")
    print("3. Run the validation to ensure your data format is correct")
    print("4. Use the training commands above to train SCHAF on your data")
    print("\nFor more information, see README.md and README_paths.md")

if __name__ == "__main__":
    main() 