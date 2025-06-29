#!/usr/bin/env python3
"""
SCHAF (Single-Cell Histology Analysis Framework)

This script provides a unified interface for both training and inference of SCHAF models.
It supports multiple scenarios including paired and unpaired training for:
- Mouse data
- Cancer data (in-sample and whole-sample)
- HTAPP data
- Placenta data
- Lung cancer data

Usage:
    python schaf_method.py --mode {train,inference} --scenario <scenario_name> [options]

Common Options:
    --gpu GPU           GPU device to use
    --fold FOLD         Fold/key to use as hold out (later for inference)
    --batch-size SIZE   Batch size for training/inference
    --workers NUM       Number of worker processes
    --pin-memory       Use pinned memory for data loading

Training-specific Options:
    --num-epochs NUM    Number of epochs to train
    --lr LR            Learning rate
    --val-size SIZE    Validation set size
    --test-size SIZE   Test set size
    --tile-radius RAD  Radius of image tiles
    --use-wandb        Enable Weights & Biases logging
    --save-model       Save model checkpoints

For detailed usage instructions, see the documentation.
"""

# Standard library imports
import os
import sys
import json
import math
import random
import argparse
from collections import defaultdict
from pathlib import Path
import datetime
from typing import Dict, List, Tuple, Set, Union, Optional

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
import torchvision
from torchvision import transforms
import wandb
import scanpy as sc
from tqdm import tqdm
import sklearn.model_selection
from PIL import Image
import seaborn as sns
from numba import njit, prange
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree
import tangram as tg
import imageio.v3 as iio
import cv2
import scgpt
import timm
from einops import rearrange
from torch import einsum
import torch.nn.utils as U

# Configure system settings
Image.MAX_IMAGE_PIXELS = 933120000  # Allow loading large images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 6 if torch.cuda.is_available() else 2
PIN_MEMORY = torch.cuda.is_available()
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

# Import path configurations
from schaf_paths import *

# Add project paths
PROJECT_ROOT = Path(os.getcwd()).parent
sys.path.extend([
    str(PROJECT_ROOT),
])

#######################################################################################
# Model Classes from models.py
#######################################################################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x

class ViT_UNI(nn.Module):
    def __init__(self, num_genes=1000, local_dir=MODEL_PATHS['vit_uni_checkpoint']):
        super(ViT_UNI, self).__init__()
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

    def forward(self, x):
        x = self.model(x)
        return x

class MerNet(nn.Module):
    def __init__(self, num_genes, model_name, pretrained, num_hidden_layers, pretrain_hist=False, pretrain_st='NONE', big_lin=False, have_relu=True):
        super(MerNet, self).__init__()
        self.model_name = model_name
        
        if model_name == 'xj_transformer':
            self.part_one = ViT_UNI()
            if pretrain_st != 'NONE':
                MODEL_PATH = pretrain_st
                def load_model_weights(model, weights):
                    model_dict = model.state_dict()
                    weights = {k: v for k, v in weights.items() if k in model_dict}
                    if weights == {}:
                        print('No weight could be loaded..')
                    else:
                        print(len(weights), ' weights were loaded here')
                    model_dict.update(weights)
                    model.load_state_dict(model_dict)
                    return model

                state = torch.load(MODEL_PATH, map_location='cuda')
                try:
                    state_dict = state['state_dict']
                except:
                    state_dict = state['model_state_dict']

                for key in list(state_dict.keys()):
                    state_dict[key.replace('resnet.', '').replace('part_one.', '')] = state_dict.pop(key)
        
                self.part_one = load_model_weights(self.part_one, state_dict)
        else:
            self.part_one = torch.hub.load('pytorch/vision:v0.11.3', model_name, pretrained=pretrained)

        the_embed_dim_use = 1000 if model_name != 'xj_transformer' else 1024
        
        # Build part_two based on parameters
        layers = []
        if not have_relu and (not big_lin) and num_hidden_layers == 0:
            layers.append(nn.Linear(the_embed_dim_use, num_genes))
        else:
            # Add initial layer
            mid = 1<<9 if num_genes < 1000 else 1<<10 if num_genes < 1<<11 else 1<<11
            layers.extend([
                nn.Linear(the_embed_dim_use, mid),
                nn.BatchNorm1d(mid),
                nn.ReLU()
            ])
            
            # Add hidden layers based on num_hidden_layers
            for i in range(1, min(num_hidden_layers, 4)):
                prev_dim = mid
                mid = 1<<(8-i) if num_genes < 1000 else 1<<10 if num_genes < 1<<11 else 1<<(11+i)
                layers.extend([
                    nn.Linear(prev_dim, mid),
                    nn.BatchNorm1d(mid),
                    nn.ReLU()
                ])
            
            # Add output layer
            layers.append(nn.Linear(mid, num_genes))
            if have_relu:
                layers.append(nn.ReLU())
                
        self.part_two = nn.Sequential(*layers)

    def forward(self, x):
        return self.part_two(self.part_one(x))

class JustPartTwo(nn.Module):
    def __init__(self, num_genes, num_hidden_layers, have_relu=True, big_lin=False, mlp_dim=4000):
        super(JustPartTwo, self).__init__()
        
        # Build network architecture
        layers = []
        if not have_relu and (not big_lin) and num_hidden_layers == 0:
            layers.append(nn.Linear(mlp_dim, num_genes))
        else:
            # Add initial layer
            mid = 1<<9 if num_genes < 1000 else 1<<10 if num_genes < 1<<11 else 1<<11
            layers.extend([
                nn.Linear(mlp_dim, mid),
                nn.BatchNorm1d(mid),
                nn.ReLU()
            ])
            
            # Add hidden layers based on num_hidden_layers
            for i in range(1, min(num_hidden_layers, 4)):
                prev_dim = mid
                mid = 1<<(8-i) if num_genes < 1000 else 1<<10 if num_genes < 1<<11 else 1<<(11+i)
                layers.extend([
                    nn.Linear(prev_dim, mid),
                    nn.BatchNorm1d(mid),
                    nn.ReLU()
                ])
            
            # Add output layer
            layers.append(nn.Linear(mid, num_genes))
            if have_relu:
                layers.append(nn.ReLU())
                
        self.part_two = nn.Sequential(*layers)

    def forward(self, x):
        return self.part_two(x)

class HEGen(nn.Module):
    def __init__(self, latent_dim=1<<9):
        super(HEGen, self).__init__()
        self.part_two = nn.Sequential(
            nn.Linear(1<<9, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<9),
            nn.BatchNorm1d(1<<9),
            nn.ReLU(),
            nn.Linear(1<<9, latent_dim),
        )
    def forward(self, x):
        return self.part_two(x)

class StandardDecoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=512, hidden_dim=2048):
        super(StandardDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.latent_dim = latent_dim
    def forward(self, x):
        res = self.net(x)
        return res

class TransferModel(nn.Module):
    def __init__(self, part1, part2):
        super(TransferModel, self).__init__()
        self.part_one = part1
        self.part_two = part2
    def forward(self, x):
        return self.part_two(self.part_one(x))

class Discriminator(nn.Module):
    def __init__(self, latent_dim=1<<9, spectral=True, end_dim=2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            U.spectral_norm(nn.Linear(latent_dim, 1<<8)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<8, 1<<7)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<7, 1<<6)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<6, 1<<5)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<5, end_dim)),
        )
    def forward(self, x):
        return self.net(x)

class HEDecoder(nn.Module):
    def __init__(self, latent_dim=1<<9):
        super(HEDecoder, self).__init__()
        self.part_two = nn.Sequential(
            nn.Linear(latent_dim, 1<<9),
            nn.BatchNorm1d(1<<9),
            nn.ReLU(),
            nn.Linear(1<<9, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<9),
        )
    def forward(self, x):
        return self.part_two(x)

#######################################################################################
# Configuration Constants
#######################################################################################

# Model architecture constants
SC_EMBEDDING_DIM = 512
HIST_EMBEDDING_DIM = 1024
HIDDEN_DIM = 2048
LATENT_DIM = 512

# Training constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VAL_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TILE_RADIUS = 112

# Dataset configurations
SCENARIO_CONFIGS = {
    'mouse': {
        'data_dir': DATA_PATHS['mouse_xenium'],
        'he_path': 'Xenium_V1_mouse_pup_he_image.ome.tif',
        'model_dir': MODEL_PATHS['mouse_models'],
        'output_dir': OUTPUT_PATHS['mouse_inferences'],
        'proj_dir': OUTPUT_PATHS['mouse_folds'],
        'stage1_suffix': 'st.h5ad',
        'stage2_suffix': 'tang_proj.h5ad',
        'use_hold_out': True,
        'model_name_prefix': 'mouse_leave_out_fold',
        'is_paired': True,
        'tile_radius': 180,
        'needs_transform': False
    },
    'cancer_in_sample': {
        'data_dir': DATA_PATHS['cancer_xenium'],
        'he_path': 'xenium_hist.png',
        'model_dir': MODEL_PATHS['xenium_cancer_models'],
        'output_dir': OUTPUT_PATHS['xenium_cancer_inferences'],
        'proj_dir': OUTPUT_PATHS['cancer_folds'],
        'stage1_suffix': 'st.h5ad',
        'stage2_suffix': 'tang_proj.h5ad',
        'use_hold_out': True,
        'model_name_prefix': 'xenium_cancer_leave_out_fold',
        'is_paired': True,
        'tile_radius': 180,
        'needs_transform': False
    },
    'cancer_whole_sample': {
        'data_dir': DATA_PATHS['cancer_xenium'],
        'he_path': 'HE_other_sample_xenium.tif',
        'model_dir': MODEL_PATHS['xenium_cancer_models'],
        'output_dir': OUTPUT_PATHS['xenium_cancer_inferences'],
        'proj_dir': OUTPUT_PATHS['cancer_folds'],
        'stage1_suffix': 'st.h5ad',
        'stage2_suffix': 'tang_proj.h5ad',
        'use_hold_out': False,
        'model_name_prefix': 'xenium_cancer_whole_sample',
        'is_paired': True,
        'tile_radius': 150,
        'needs_transform': True,
        'transform_file': 'alignment_new_xen.csv'
    },
    'htapp': {
        'save_models_dir': MODEL_PATHS['htapp_models'],
        'hist_embeddings_dir': EMBEDDING_PATHS['htapp_hist_embeddings'],
        'sc_embeddings_dir': EMBEDDING_PATHS['htapp_sc_embeddings'],
        'sc_dir': DATA_PATHS['htapp_sc'],
        'keys': ['4531', '7179', '7479', '7629', '932', '6760', '7149', '4381', '8239'],
        'is_paired': False,
        'hist_embed_shape': 1024,
        'sc_embed_shape': 512,
        'output_dir': OUTPUT_PATHS['htapp_inferences']
    },
    'placenta': {
        'save_models_dir': MODEL_PATHS['placenta_models'],
        'hist_embeddings_dir': EMBEDDING_PATHS['placenta_hist_embeddings'],
        'sc_embeddings_dir': EMBEDDING_PATHS['placenta_sc_embeddings'],
        'sc_dir': DATA_PATHS['placenta_sc'],
        'keys': ['7', '8', '9', '11'],
        'k_to_code': {'7': 'JS34', '8': 'JS40', '9': 'JS35', '11': 'JS36'},
        'is_paired': False,
        'output_dir': OUTPUT_PATHS['placenta_inferences']
    },
    'lung_cancer': {
        'save_models_dir': MODEL_PATHS['lung_cancer_models'],
        'hist_embeddings_dir': EMBEDDING_PATHS['lung_cancer_hist_embeddings'],
        'sc_embeddings_dir': EMBEDDING_PATHS['lung_cancer_sc_embeddings'],
        'sc_dir': DATA_PATHS['lung_cancer_sc'],
        'is_paired': False,
        'hist_embed_shape': 1024,
        'sc_embed_shape': 512,
        'output_dir': OUTPUT_PATHS['lung_cancer_inferences']
    }
}

# Pre-training data preparation constants
MOUSE_XEN_DIR = DATA_PATHS['mouse_xenium']
CANCER_XEN_DIR = DATA_PATHS['cancer_xenium']
MOUSE_CHUNKS_DIR = TEMP_PATHS['mouse_chunks']
CANCER_CHUNKS_DIR = TEMP_PATHS['cancer_chunks']
CORRELATION_THRESHOLD = 0.5
CHUNK_SIZE = 22500
MOUSE_SAMPLE_SIZE = 45000

# Training preparation configurations
PREP_CONFIGS = {
    'mouse': {
        'data_dir': DATA_PATHS['mouse_xenium'],
        'chunks_dir': TEMP_PATHS['mouse_chunks_alt'],
        'output_dir': OUTPUT_PATHS['mouse_folds'],
        'num_chunks': 60,
        'transform_coords': True,
        'zone_boundaries': {
            'y_split': 19500,
            'x_split': 36500
        }
    },
    'cancer': {
        'data_dir': DATA_PATHS['cancer_xenium'],
        'chunks_dir': TEMP_PATHS['cancer_chunks'],
        'output_dir': OUTPUT_PATHS['cancer_folds'],
        'num_chunks': 8,
        'transform_coords': False,
        'zone_boundaries': {
            'y_split': 12900,
            'x_split': 17700
        }
    }
}

#######################################################################################
# Dataset Classes
#######################################################################################

class ImageDataset(Dataset):
    """Base dataset class for image-based training and inference"""
    def __init__(self, he_image: np.ndarray, adata: sc.AnnData, tile_radius: int, indices: np.ndarray):
        self.he_image = he_image
        self.adata = adata
        self.tile_radius = tile_radius
        self.indices = indices
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[index]
        x = int(self.adata.obs['x'][index])
        y = int(self.adata.obs['y'][index])
        
        img = np.zeros((self.tile_radius*2, self.tile_radius*2, 3), dtype=np.float32)
        to_be = self.he_image[
            max(0, y-self.tile_radius):max(0, y+self.tile_radius),
            max(0, x-self.tile_radius):max(0, x+self.tile_radius),
        ]

        img[:to_be.shape[0], :to_be.shape[1]] = to_be

        if img.max() > 1.:
            img = img / 255.

        trans = self.adata.X[index].astype(np.float32)
        img = self.transforms(img)

        return img, trans

class UnpairedDataset(Dataset):
    """Dataset class for unpaired training scenarios"""
    def __init__(self, embeddings: np.ndarray, is_hist: bool = True):
        self.embeddings = embeddings
        self.is_hist = is_hist
        
    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.embeddings[idx]

#######################################################################################
# Training Functions
#######################################################################################
# Common dataset class for histology embeddings
class HistSampleDataset(Dataset):
    def __init__(self, he_image, xs, ys, tile_radius):
        self.he_image = he_image
        self.xs = xs
        self.ys = ys
        self.tile_radius = tile_radius
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.trnsfrms_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, index):
        x = int(self.xs[index])
        y = int(self.ys[index])
        img = np.zeros((self.tile_radius*2, self.tile_radius*2, 3), dtype=np.float32)
        to_be = self.he_image[
            max(0, y-self.tile_radius):max(0, y+self.tile_radius),
            max(0, x-self.tile_radius):max(0, x+self.tile_radius),
        ]
        img[:to_be.shape[0],:to_be.shape[1]] = to_be
        if img.max() > 1.:
            img = img / 255.
        img = self.trnsfrms_val(img)
        return img


# Add transform_coordinates definition (from old_schaf_inference.py)
from numba import njit, prange
import numpy as np

def transform_coordinates(horis: np.ndarray, verts: np.ndarray, inv_trans: np.ndarray) -> tuple:
    new_horis, new_verts = np.zeros_like(horis), np.zeros_like(verts)
    for t in range(len(horis)):
        i = horis[t] / .2125
        j = verts[t] / .2125
        new = inv_trans.dot(np.array([i, j, 1]))
        new_horis[t] = new[0]
        new_verts[t] = new[1]
    return new_horis, new_verts

device = torch.device("cuda")
def generate_embeddings(scenario: str, config: dict, modality: str = 'both') -> None:
    """
    Generate embeddings for different scenarios using a unified approach.
    This combines the embedding generation from htapp, placenta, and lung cancer scripts.
    
    Args:
        scenario (str): The scenario to generate embeddings for ('htapp', 'placenta', or 'lung_cancer')
        config (dict): Configuration dictionary for the scenario
        modality (str): Which embeddings to generate - 'hist', 'sc', or 'both' (default)
    """
    device = torch.device("cuda")

    # Set tile_radius per scenario
    if scenario == 'htapp':
        tile_radius = 75
    elif scenario in ['lung_cancer', 'placenta']:
        tile_radius = 210
    else:
        tile_radius = 210

    if modality in ['hist', 'both']:
        # Initialize histology embedding model
        embedding_maker = ViT_UNI().to(device)
        embedding_maker = embedding_maker.eval()

        if scenario == 'htapp':
            # Process HTAPP histology data
            hists_dir = DATA_PATHS['htapp_hists']
            for f in os.listdir(hists_dir):
                if '.png' not in f:
                    continue
                k = f.split('.')[0]
                hist = iio.imread(os.path.join(hists_dir, f))
                seg = pd.read_csv(f'{DATA_PATHS["htapp_hist_info"]}/{k}_info.txt', sep="\t", header=0)
                xs = np.array(seg['Centroid X px']).astype(int)
                ys = np.array(seg['Centroid Y px']).astype(int)
                
                process_and_save_embeddings(hist, xs, ys, k, embedding_maker, config['hist_embeddings_dir'], tile_radius)

        elif scenario == 'placenta':
            # Process placenta histology data
            k_to_code = config['k_to_code']
            for k in k_to_code:
                name = k_to_code[k]
                image = cv2.imread(f'{DATA_PATHS["placenta_hists"]}/{name}.jpg', -1)
                xs = np.load(f'{DATA_PATHS["placenta_hists"]}/{name}_final_xs.npy')
                ys = np.load(f'{DATA_PATHS["placenta_hists"]}/{name}_final_ys.npy')
                
                process_and_save_embeddings(image, xs, ys, k, embedding_maker, config['hist_embeddings_dir'], tile_radius)

        elif scenario == 'lung_cancer':
            # Process lung cancer histology data
            the_dir = DATA_PATHS['lung_cancer_segmentation']
            for f in os.listdir(the_dir):
                if 'chunk' not in f:
                    continue
                name = f.split('.')[0]
                other_f = f.replace('chunk', 'seg').replace('jpg', 'csv')
                the_chunk = iio.imread(os.path.join(the_dir, f))
                the_seg = pd.read_csv(os.path.join(the_dir, other_f))
                xs = np.array(the_seg['centroid_y']).astype(int)
                ys = np.array(the_seg['centroid_x']).astype(int)
                
                process_and_save_embeddings(the_chunk, xs, ys, name, embedding_maker, config['hist_embeddings_dir'], tile_radius)

        else:  # this is for the demo case 
            xs = np.array([i for i in range(100, 600)]) 
            ys = np.array([i for i in range(100, 600)])
            process_and_save_embeddings(the_chunk, xs, ys, name, embedding_maker, config['hist_embeddings_dir'], tile_radius)



    if modality in ['sc', 'both']:
        # Initialize scRNA-seq embedding model
        model_dir = MODEL_PATHS['scgpt_pancancer']
        
        if scenario == 'htapp':
            # Process HTAPP scRNA-seq data
            for key in config['keys']:
                sc_adata = sc.read_h5ad(f"{config['sc_dir']}/{key}.h5ad")
                new_v = sc.AnnData(X=np.array(sc_adata.obsm['counts'].todense()), obs=sc_adata.obs)
                sc.pp.log1p(new_v)
                new_v.var.index = sc_adata.uns['counts_var']
                sc_adata = new_v
                sc.pp.filter_cells(sc_adata, min_genes=1)
                sc.pp.filter_genes(sc_adata, min_cells=1)
                sc_adata.var['gene_col'] = list(sc_adata.var.index)
                
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    embed_data = scgpt.tasks.embed_data(sc_adata, model_dir, gene_col='gene_col', batch_size=64)
                    new_adata = sc.AnnData(X=embed_data.obsm['X_scGPT'], obs=embed_data.obs)
                    os.makedirs(config['sc_embeddings_dir'], exist_ok=True)
                    new_adata.write(f"{config['sc_embeddings_dir']}/{key}.h5ad")

        elif scenario == 'placenta':
            # Process placenta scRNA-seq data
            model_dir = MODEL_PATHS['scgpt_human']
            the_sc = sc.read_h5ad(DATA_PATHS['placenta_metadata'])
            
            for k, code in config['k_to_code'].items():
                sc_adata = the_sc[the_sc.obs['Sample']==code]
                sc_adata.var['gene_col'] = list(sc_adata.var.index)
                sc_adata.X = np.array(sc_adata.X.todense())
                sc.pp.log1p(sc_adata)
                sc.pp.filter_cells(sc_adata, min_genes=1)
                sc.pp.filter_genes(sc_adata, min_cells=1)
                
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    embed_data = scgpt.tasks.embed_data(sc_adata, model_dir, gene_col='gene_col', batch_size=64)
                    new_adata = sc.AnnData(X=embed_data.obsm['X_scGPT'], obs=embed_data.obs)
                    os.makedirs(config['sc_embeddings_dir'], exist_ok=True)
                    new_adata.write(f"{config['sc_embeddings_dir']}/{k}.h5ad")

        elif scenario == 'lung_cancer':
            # Process lung cancer scRNA-seq data
            for file in os.listdir(config['sc_dir']):
                if not file.endswith('.h5ad'):
                    continue
                k = file.split('.')[0]
                sc_adata = sc.read_h5ad(os.path.join(config['sc_dir'], file))
                sc_adata.var['gene_col'] = list(sc_adata.var.index)
                sc.pp.log1p(sc_adata)
                sc.pp.filter_cells(sc_adata, min_genes=1)
                sc.pp.filter_genes(sc_adata, min_cells=1)
                
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    embed_data = scgpt.tasks.embed_data(sc_adata, model_dir, gene_col='gene_col', batch_size=64)
                    new_adata = sc.AnnData(X=embed_data.obsm['X_scGPT'], obs=embed_data.obs)
                    os.makedirs(config['sc_embeddings_dir'], exist_ok=True)
                    new_adata.write(f"{config['sc_embeddings_dir']}/{k}.h5ad")

        else:  # demo example
            # Process placenta scRNA-seq data
            model_dir = MODEL_PATHS['scgpt_human']
            the_sc = sc.read_h5ad(CUSTOM_DATA_PATHS['custom_sc_data'])
            
            for k in range(4):
                sc_adata = the_sc[the_sc.obs['sample']==code]
                sc_adata.var['gene_col'] = list(sc_adata.var.index)
                sc_adata.X = np.array(sc_adata.X)
                sc.pp.log1p(sc_adata)
                sc.pp.filter_cells(sc_adata, min_genes=1)
                sc.pp.filter_genes(sc_adata, min_cells=1)
                
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    embed_data = scgpt.tasks.embed_data(sc_adata, model_dir, gene_col='gene_col', batch_size=64)
                    new_adata = sc.AnnData(X=embed_data.obsm['X_scGPT'], obs=embed_data.obs)
                    os.makedirs(config['sc_embeddings_dir'], exist_ok=True)
                    new_adata.write(f"{config['sc_embeddings_dir']}/{k}.h5ad")

def process_and_save_embeddings(image, xs, ys, name, embedding_maker, save_dir, tile_radius):
    """Helper function to process and save embeddings for a single image."""
    the_ds = HistSampleDataset(image, xs, ys, tile_radius)
    the_dl = DataLoader(the_ds, 64, shuffle=0, num_workers=6, pin_memory=1)
    
    with torch.no_grad():
        embeds = []
        for batch in tqdm(the_dl):
            res = embedding_maker(batch.to(device).float()).cpu().detach().numpy()
            embeds.extend(res)    
        torch.cuda.empty_cache()

    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/{name}.npy', np.stack(embeds))

def prepare_paired_data(scenario: str, force_recompute: bool = False, custom_config: dict = None) -> None:
    """
    Prepare data for paired training scenarios (mouse, cancer, and custom).
    This combines the functionality from schaf_before_training.py.
    """
    if scenario == 'custom':
        if not custom_config:
            raise ValueError("custom_config is required for custom scenarios")
        config = custom_config
    else:
        config = PREP_CONFIGS[scenario]
    
    # Check if data already exists and force_recompute is False
    if not force_recompute:
        output_files_exist = all(
            os.path.exists(os.path.join(config['output_dir'], f'fold_{z}_{suffix}.h5ad'))
            for z in range(4)
            for suffix in ['st', 'tang_proj']
        )
        if output_files_exist:
            print(f"Data for {scenario} already exists, skipping preparation...")
            return
    
    print(f"Preparing {scenario} data...")
    
    # Run Tangram processing
    if scenario == 'custom':
        train_genes = run_tangram(scenario, random_split=True, custom_config=custom_config)
    else:
        train_genes = run_tangram(scenario, random_split=True)
    
    # Load and prepare data
    if scenario == 'mouse':
        st_data, metadata = load_mouse_prep_data(config, train_genes)
    elif scenario == 'custom':
        st_data, metadata = load_custom_prep_data(config, train_genes)
    else:
        st_data, metadata = load_cancer_prep_data(config, train_genes)
    
    # Process chunks and save results
    sts, tang_projs = process_chunks_prep(config, st_data, metadata, scenario)
    concatenate_and_save(sts, tang_projs, config, force_recompute)
    
    print(f"Finished preparing {scenario} data!")

def get_zone(x: int, y: int, boundaries: Dict[str, int]) -> int:
    """Determine which zone a point belongs to based on its coordinates."""
    if (y < boundaries['y_split'] and x < boundaries['x_split']):
        return 0
    elif (y >= boundaries['y_split'] and x >= boundaries['x_split']):
        return 1
    elif (y < boundaries['y_split'] and x >= boundaries['x_split']):
        return 2
    else:
        return 3


def load_mouse_prep_data(config: Dict, train_genes: Set[str]) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Load and preprocess mouse data for training preparation."""
    # Load spatial transcriptomics data
    st_data = sc.read_10x_h5(os.path.join(config['data_dir'], 'cell_feature_matrix.h5'))
    
    # Load metadata
    metadata = pd.read_csv(os.path.join(config['data_dir'], 'cells.csv'))
    metadata = metadata.set_index('cell_id')
    
    # Load and apply coordinate transformation
    horis = np.array(metadata['x_centroid'])
    verts = np.array(metadata['y_centroid'])
    
    df = pd.read_csv(os.path.join(config['data_dir'], 'Xenium_V1_mouse_pup_he_imagealignment.csv'), header=None)
    transformation_matrix = df.values.astype(np.float32)
    inv_trans = np.linalg.inv(transformation_matrix).astype('float64')
    
    new_horis, new_verts = transform_coordinates(horis, verts, inv_trans)
    xs = new_horis.astype(int)
    ys = new_verts.astype(int)
    
    # Add zone information
    xenium_zone = np.array([
        int(get_zone(x, y, config['zone_boundaries'])) 
        for x, y in zip(xs, ys)
    ])
    
    # Update metadata
    metadata['zone'] = xenium_zone
    metadata['x'] = xs
    metadata['y'] = ys
    
    # Clean up metadata
    columns_to_drop = [
        'x_centroid', 'y_centroid', 'transcript_counts', 'control_probe_counts',
        'control_codeword_counts', 'unassigned_codeword_counts', 
        'deprecated_codeword_counts', 'total_counts', 'cell_area', 'nucleus_area'
    ]
    metadata = metadata.drop(columns=columns_to_drop)
    
    # Load and add clustering information
    broad_clusters = pd.read_csv(
        os.path.join(config['data_dir'], 'analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv')
    ).set_index('Barcode')
    
    fine_clusters = pd.read_csv(
        os.path.join(config['data_dir'], 'analysis/clustering/gene_expression_graphclust/clusters.csv')
    ).set_index('Barcode')
    
    metadata = metadata.loc[fine_clusters.index]
    metadata['broad_clusters'] = broad_clusters.loc[metadata.index]['Cluster']
    metadata['fine_clusters'] = fine_clusters.loc[metadata.index]['Cluster']
    
    return st_data[::,list(train_genes)], metadata

def load_cancer_prep_data(config: Dict, train_genes: Set[str]) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Load and preprocess cancer data for training preparation."""
    # Load data
    st_data = sc.read_h5ad(os.path.join(config['data_dir'], 'xenium_breast.h5ad'))
    st_data.var.index = [g.lower() for g in st_data.var.index]
    
    # Load clustering information
    broad_clusters = pd.read_csv(
        f'{DATA_PATHS["cancer_clustering_base"]}/gene_expression_kmeans_10_clusters/clusters.csv'
    )['Cluster']
    
    fine_clusters = pd.read_csv(
        f'{DATA_PATHS["cancer_clustering_base"]}/gene_expression_graphclust/clusters.csv'
    )['Cluster']
    
    # Get coordinates
    xs = st_data.obsm['spatial'][:, 0].astype(int)
    ys = st_data.obsm['spatial'][:, 1].astype(int)
    
    # Create metadata
    metadata = pd.DataFrame(index=st_data.obs.index)
    metadata['x'] = xs
    metadata['y'] = ys
    metadata['broad_clusters'] = broad_clusters
    metadata['fine_clusters'] = fine_clusters
    metadata['zone'] = [
        get_zone(x, y, config['zone_boundaries']) 
        for x, y in zip(xs, ys)
    ]
    
    # Clean up AnnData object
    del st_data.obsm
    del st_data.obs['n_genes']
    del st_data.var['gene_ids']
    del st_data.var['feature_types']
    del st_data.var['genome']
    
    return st_data[::,list(train_genes)], metadata

def load_custom_prep_data(config: Dict, train_genes: Set[str]) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Load and preprocess custom user data for training preparation."""
    # Load spatial transcriptomics data
    st_data = sc.read_h5ad(os.path.join(config['data_dir'], config['st_file']))
    
    # Ensure spatial coordinates are available
    if 'x' in st_data.obs and 'y' in st_data.obs:
        xs = st_data.obs['x'].astype(int)
        ys = st_data.obs['y'].astype(int)
    elif 'spatial' in st_data.obsm:
        xs = st_data.obsm['spatial'][:, 0].astype(int)
        ys = st_data.obsm['spatial'][:, 1].astype(int)
    else:
        raise ValueError("Spatial coordinates not found. Expected in obsm['spatial'] or obs['x']/obs['y']")
    
    # Create metadata
    metadata = pd.DataFrame(index=st_data.obs.index)
    metadata['x'] = xs
    metadata['y'] = ys
    
    # Add zone information (simple quadrant-based zoning)
    x_median = np.median(xs)
    y_median = np.median(ys)
    
    zones = []
    for x, y in zip(xs, ys):
        if x < x_median and y < y_median:
            zone = 0
        elif x >= x_median and y >= y_median:
            zone = 1
        elif x < x_median and y >= y_median:
            zone = 2
        else:
            zone = 3
        zones.append(zone)
    
    metadata['zone'] = zones
    
    # Add clustering information if available
    if 'cluster' in st_data.obs:
        metadata['broad_clusters'] = st_data.obs['cluster']
        metadata['fine_clusters'] = st_data.obs['cluster']
    else:
        # Create simple clustering based on spatial location
        metadata['broad_clusters'] = metadata['zone']
        metadata['fine_clusters'] = metadata['zone']
    
    return st_data[::,list(train_genes)], metadata

def process_chunks_prep(config: Dict, 
                       st_data: sc.AnnData, 
                       metadata: pd.DataFrame,
                       dataset: str = 'custom') -> Tuple[Dict, Dict]:
    """Process data chunks and organize them by zone."""
    sts = defaultdict(list)
    tang_projs = defaultdict(list)
    
    # Calculate chunk sizes and indices
    chunk_size = CHUNK_SIZE  # Same as CHUNK_SIZE used in process_multiple_chunks
    n_spots = st_data.shape[0]
    
    # Check which chunk files actually exist
    existing_chunks = []
    for i in range(config['num_chunks']):
        proj_path = os.path.join(config['chunks_dir'], f'{dataset}_random_chunk_{i}.h5ad')
        if os.path.exists(proj_path):
            existing_chunks.append(i)
    
    if not existing_chunks:
        raise ValueError(f"No chunk files found in {config['chunks_dir']}")
    
    print(f"Found {len(existing_chunks)} chunk files: {existing_chunks}")
    
    for i in tqdm(existing_chunks, desc='Processing chunks'):
        # Load projection data - use the correct naming pattern from process_multiple_chunks
        proj_path = os.path.join(config['chunks_dir'], f'{dataset}_chunk_{i}.h5ad')
        the_proj = sc.read_h5ad(proj_path)
        
        # Clean up projection data - use try-except to handle missing columns
        columns_to_remove_obs = ['n_genes', 'uniform_density', 'rna_count_based_density']
        for col in columns_to_remove_obs:
            try:
                del the_proj.obs[col]
            except KeyError:
                pass  # Column doesn't exist, skip it
        
        # Clean up var columns
        var_columns_to_remove = ['gene_ids', 'feature_types', 'genome', 'gene_id', 'gene_type', 'chr', 'n_cells', 'is_training']
        for col in var_columns_to_remove:
            try:
                del the_proj.var[col]
            except KeyError:
                pass  # Column doesn't exist, skip it
        
        # Calculate spatial indices for this chunk (matching process_multiple_chunks logic)
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_spots)
        spatial_indices = st_data.obs.index[start_idx:end_idx]
        
        # Get corresponding ST data and metadata using spatial indices
        this_st = st_data[spatial_indices]
        this_metadata = metadata.loc[spatial_indices]
        
        # For now, skip zone processing and just process by chunks
        # Add chunk identifier and process as a single unit
        this_st.obs = this_metadata
        the_proj.obs['chunk'] = i
        
        # The projection data contains single-cell observations mapped to spatial locations
        # We need to add spatial coordinates based on the mapping from Tangram
        # For now, we'll handle this in the concatenate_and_save function
        
        # Add to zone 0 for simplicity (we can enhance zone processing later)
        sts[0].append(this_st)
        tang_projs[0].append(the_proj)
    
    return sts, tang_projs

def concatenate_and_save(sts: Dict, 
                        tang_projs: Dict, 
                        config: Dict,
                        force_recompute: bool = False) -> None:
    """Concatenate data by zone and save to files."""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    for z in range(4):
        # Skip zones that don't have any data
        if len(sts[z]) == 0 or len(tang_projs[z]) == 0:
            print(f'Zone {z} has no data, skipping...')
            continue
            
        st_path = os.path.join(config['output_dir'], f'fold_{z}_st.h5ad')
        proj_path = os.path.join(config['output_dir'], f'fold_{z}_tang_proj.h5ad')
        
        if not force_recompute and os.path.exists(st_path) and os.path.exists(proj_path):
            print(f'Files for zone {z} already exist, skipping...')
            continue
            
        print(f'Processing zone {z}...')
        
        # Concatenate ST data
        if len(sts[z]) == 1:
            zone_st = sts[z][0]
        else:
            zone_st = sts[z][0].concatenate(
                *[sts[z][i] for i in range(len(sts[z])) if i > 0]
            )
        zone_st.var.index = [g.lower() for g in zone_st.var.index]
        
        # Concatenate projection data
        if len(tang_projs[z]) == 1:
            zone_proj = tang_projs[z][0]
        else:
            zone_proj = tang_projs[z][0].concatenate(
                *[tang_projs[z][i] for i in range(len(tang_projs[z])) if i > 0]
            )
        zone_proj.var.index = [g.lower() for g in zone_proj.var.index]
        
        # Ensure projection data has the same genes as ST data (training genes)
        # This is critical for stage 2 training compatibility
        # if zone_st.shape[1] != zone_proj.shape[1]:
        #     # Get the training genes from ST data
        #     st_genes = zone_st.var.index
        #     # Filter projection data to match ST genes
        #     common_genes = list(set(st_genes) & set(zone_proj.var.index))
        #     if len(common_genes) == len(st_genes):
        #         # If all ST genes are in projection, subset to ST genes in same order
        #         zone_proj = zone_proj[:, st_genes]
        #     else:
        #         print(f"Warning: Only {len(common_genes)}/{len(st_genes)} genes overlap between ST and projection data")
        #         zone_proj = zone_proj[:, common_genes]
        
        # Handle spatial coordinates for projection data
        # The projection data contains single-cell observations mapped to spatial locations
        # We need to extract spatial coordinates from the mapping
        if len(zone_proj.obs.columns) > 0:
            # Create dummy coordinates for projection data based on available spatial locations
            # This is a simplified approach - in practice, Tangram provides spatial mapping
            n_proj_obs = zone_proj.shape[0]
            n_spatial_obs = zone_st.shape[0]
            
            # Simple approach: assign coordinates by cycling through available spatial locations
            if 'x' in zone_st.obs and 'y' in zone_st.obs:
                x_coords = zone_st.obs['x'].values
                y_coords = zone_st.obs['y'].values
                
                # Cycle through spatial coordinates for all projection observations
                zone_proj.obs['x'] = [x_coords[i % len(x_coords)] for i in range(n_proj_obs)]
                zone_proj.obs['y'] = [y_coords[i % len(y_coords)] for i in range(n_proj_obs)]
        
        # Save files
        zone_st.write(st_path)
        zone_proj.write(proj_path)
        print(f'Saved files for zone {z}')

def normalize_data(fold_to_trans: dict, 
                  hold_out_fold: str = None) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Normalize transcriptomics data across all folds except the hold-out fold.
    
    This function computes mean and standard deviation across all training folds
    and applies the normalization to all folds (including hold-out).
    
    Args:
        fold_to_trans (dict): Dictionary mapping fold IDs to AnnData objects
        hold_out_fold (str, optional): Fold ID to exclude from normalization statistics
        
    Returns:
        tuple:
            - dict: Normalized fold_to_trans dictionary
            - np.ndarray: Mean values used for normalization
            - np.ndarray: Standard deviation values used for normalization
    """
    # Get dimensions
    num_genes = next(iter(fold_to_trans.values())).shape[1]
    
    # Calculate total samples excluding hold-out fold
    total_samples = sum(
        v.shape[0] for k, v in fold_to_trans.items() 
        if hold_out_fold is None or k != hold_out_fold
    )
    
    # Calculate weighted mean across all training folds
    all_means = sum(
        v.shape[0] * v.X.mean(axis=0) 
        for k, v in fold_to_trans.items() 
        if hold_out_fold is None or k != hold_out_fold
    ) / total_samples
    
    # Calculate weighted variance across all training folds
    all_vars = sum(
        v.shape[0] * (v.X.var(axis=0) + np.square(v.X.mean(axis=0) - all_means))
        for k, v in fold_to_trans.items()
        if hold_out_fold is None or k != hold_out_fold
    ) / total_samples
    
    # Calculate standard deviation
    all_stds = np.sqrt(all_vars)
    
    # Apply normalization to all folds
    for fold_id, adata in fold_to_trans.items():
        fold_to_trans[fold_id].X = np.nan_to_num((adata.X - all_means) / all_stds)
    
    return fold_to_trans, all_means, all_stds

# Type aliases for complex return types
from typing import Any

def load_data_for_scenario(config: dict, args: dict, stage2: bool = False) -> Any:
    """
    Load and prepare data based on the training scenario.
    
    Args:
        config (dict): Scenario configuration
        args (dict): Training arguments
        
    Returns:
        For paired scenarios:
            tuple: (image, fold_data, mean, std)
                - image: np.ndarray - Histology image
                - fold_data: dict - Dictionary of fold data
                - mean: np.ndarray - Mean values for normalization
                - std: np.ndarray - Standard deviation values for normalization
        For unpaired scenarios:
            tuple: (hist_embeddings, sc_embeddings)
                - hist_embeddings: np.ndarray - Histology embeddings
                - sc_embeddings: np.ndarray - Single-cell embeddings
    """
    if config['is_paired']:
        # Load histology image
        he_image = np.array(Image.open(config['he_path']))
        
        # Load fold data from output directory where they were created
        fold_to_trans = {}
        fold_dir = config['output_dir']  # Look in output directory, not data directory
        
        print(f"Looking for fold data in: {fold_dir}")
        print(f"Looking for files ending with: {config['stage2_suffix' if stage2 else 'stage1_suffix']}")
        
        if os.path.exists(fold_dir):
            available_files = os.listdir(fold_dir)
            print(f"Available files: {available_files}")
            
            for fold in available_files:
                if not fold.endswith(config['stage2_suffix' if stage2 else 'stage1_suffix']):
                    continue
                    
                print(f"Found matching file: {fold}")
                fold_id = fold.split('_')[1]  # Extract fold number from "fold_0_st.h5ad" format
                fold_to_trans[fold_id] = sc.read_h5ad(
                    os.path.join(fold_dir, fold)
                )
        else:
            print(f"Fold directory does not exist: {fold_dir}")
            
        print(f"Loaded {len(fold_to_trans)} fold files: {list(fold_to_trans.keys())}")
        
        # Handle empty fold_to_trans case
        if len(fold_to_trans) == 0:
            raise ValueError(f"No fold data files found in {fold_dir}. Expected files ending with '{config['stage2_suffix' if stage2 else 'stage1_suffix']}'.")
        
        # Normalize data
        fold_to_trans, mu, sigma = normalize_data(
            fold_to_trans,
            args['fold'] if config['use_hold_out'] else None
        )
        
        return he_image, fold_to_trans, mu, sigma
    else:
        # Load pre-computed embeddings
        hist_embeddings = load_embeddings('hist', config)
        sc_embeddings = load_embeddings('sc', config)
        
        # Normalize embeddings
        hist_embeddings, sc_embeddings = normalize_embeddings(
            hist_embeddings, sc_embeddings, args['fold']
        )
        
        return hist_embeddings, sc_embeddings

def prepare_paired_dataloaders(fold_to_trans: dict,
                             hold_out_fold: str,
                             he_image: np.ndarray,
                             args: dict,
                             mu: np.ndarray = None,
                             sigma: np.ndarray = None) -> Any:
    """
    Prepare train and validation dataloaders for paired training.
    
    Args:
        fold_to_trans (dict): Dictionary mapping fold IDs to AnnData objects
        hold_out_fold (str): Fold ID to use as hold-out set
        he_image (np.ndarray): H&E histology image array
        args (dict): Training arguments including batch size and validation split
        mu (np.ndarray, optional): Mean values for normalization
        sigma (np.ndarray, optional): Standard deviation values for normalization
        
    Returns:
        tuple:
            - DataLoader: Training data loader
            - DataLoader: Validation data loader
    """
    fold_datasets = {}
    
    # Prepare datasets for each fold (excluding hold-out)
    for fold_id, adata in fold_to_trans.items():
        if hold_out_fold is not None and fold_id == hold_out_fold:
            continue
            
        # Split indices for train/val
        indices = np.arange(adata.shape[0])
        train_indices, val_indices = sklearn.model_selection.train_test_split(
            indices, 
            test_size=args['val_size'],
            random_state=11
        )
        
        # Create train/val datasets
        fold_datasets[fold_id] = {
            'train': ImageDataset(
                he_image=he_image,
                adata=adata,
                tile_radius=args['tile_radius'],
                indices=train_indices
            ),
            'val': ImageDataset(
                he_image=he_image,
                adata=adata,
                tile_radius=args['tile_radius'],
                indices=val_indices
            )
        }
    
    # Combine datasets from all folds
    train_dataset = ConcatDataset([ds['train'] for ds in fold_datasets.values()])
    val_dataset = ConcatDataset([ds['val'] for ds in fold_datasets.values()])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader

def prepare_unpaired_dataloaders(hist_embeddings: np.ndarray,
                               sc_embeddings: np.ndarray,
                               args: dict) -> Any:
    """
    Prepare dataloaders for unpaired training using pre-computed embeddings.
    
    Args:
        hist_embeddings (np.ndarray): Pre-computed histology embeddings
        sc_embeddings (np.ndarray): Pre-computed single-cell embeddings
        args (dict): Training arguments including batch size
        
    Returns:
        tuple:
            - DataLoader: Histology embeddings loader
            - DataLoader: Single-cell embeddings loader
    """
    # Create datasets
    hist_dataset = UnpairedDataset(hist_embeddings, is_hist=True)
    sc_dataset = UnpairedDataset(sc_embeddings, is_hist=False)
    
    # Create dataloaders
    hist_loader = DataLoader(
        hist_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    sc_loader = DataLoader(
        sc_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return hist_loader, sc_loader

def train_paired_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      optimizer: optim.Optimizer,
                      criterion: nn.Module,
                      device: torch.device,
                      args: dict) -> Any:
    """Train a paired SCHAF model.

    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (optim.Optimizer): Optimizer instance
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on
        args (dict): Training arguments

    Returns:
        Any: A tuple containing (train_losses, val_losses) where:
            train_losses (list): Training losses per epoch
            val_losses (list): Validation losses per epoch
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args['num_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, transcripts in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args["num_epochs"]}'):
            images = images.to(device)
            transcripts = transcripts.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, transcripts)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for images, transcripts in val_loader:
                images = images.to(device)
                transcripts = transcripts.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, transcripts)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Save best model
        if args['save_model'] and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_epoch_{epoch+1}.pth')
        
        # Log metrics
        if args['use_wandb']:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            })
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def train_unpaired_model(hist_loader: DataLoader,
                        sc_loader: DataLoader,
                        models: dict,
                        optimizers: dict,
                        device: torch.device,
                        args: dict) -> Any:
    """
    Train an unpaired SCHAF model using adversarial training.
    
    Args:
        hist_loader (DataLoader): Histology embeddings loader
        sc_loader (DataLoader): Single-cell embeddings loader
        models (dict): Dictionary containing all model components
        optimizers (dict): Dictionary containing all optimizers
        device (torch.device): Device to train on
        args (dict): Training arguments
        
    Returns:
        tuple:
            - dict: Training metrics history
            - dict: Best models state dict
    """
    # Initialize metrics tracking
    metrics_history = {
        'gen_losses': [], 'disc_losses': [],
        'hist_recon_losses': [], 'sc_recon_losses': []
    }
    if args['use_cell_types']:
        metrics_history['ct_losses'] = []
        metrics_history['ct_pretrain_losses'] = []
        
        # Pre-train cell type classifier
        print("Pre-training cell type classifier...")
        ct_classifier = models['ct_classifier'].to(device)
        ct_optimizer = optimizers['ct_classifier']
        
        for epoch in range(args['ct_pretrain_epochs']):
            epoch_ct_loss = 0.0
            num_ct_batches = 0
            
            for hist_batch, hist_ct in hist_loader:
                hist_batch, hist_ct = hist_batch.to(device), hist_ct.to(device)
                
                ct_optimizer.zero_grad()
                ct_logits = ct_classifier(hist_batch)
                ct_loss = F.cross_entropy(ct_logits, hist_ct)
                ct_loss.backward()
                ct_optimizer.step()
                
                epoch_ct_loss += ct_loss.item()
                num_ct_batches += 1
            
            avg_ct_loss = epoch_ct_loss / num_ct_batches
            metrics_history['ct_pretrain_losses'].append(avg_ct_loss)
            print(f'CT Pre-train Epoch {epoch+1}: Loss = {avg_ct_loss:.4f}')
            
            if args['use_wandb']:
                wandb.log({
                    'ct_pretrain_loss': avg_ct_loss,
                    'ct_pretrain_epoch': epoch
                })
    
    # Get models and optimizers
    generator = models['generator'].to(device)
    discriminator = models['discriminator'].to(device)
    hist_decoder = models['hist_decoder'].to(device)
    sc_decoder = models['sc_decoder'].to(device)
    
    gen_optimizer = optimizers['generator']
    disc_optimizer = optimizers['discriminator']
    hist_dec_optimizer = optimizers['hist_decoder']
    sc_dec_optimizer = optimizers['sc_decoder']
    
    # Training loop
    for epoch in range(args['num_epochs']):
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Create iterators for both loaders
        hist_iter = iter(hist_loader)
        sc_iter = iter(sc_loader)
        
        pbar = tqdm(range(min(len(hist_loader), len(sc_loader))),
                   desc=f'Epoch {epoch+1}/{args["num_epochs"]}')
        
        for _ in pbar:
            try:
                if args['use_cell_types']:
                    hist_batch, hist_ct = next(hist_iter)
                    sc_batch, sc_ct = next(sc_iter)
                    hist_batch, hist_ct = hist_batch.to(device), hist_ct.to(device)
                    sc_batch, sc_ct = sc_batch.to(device), sc_ct.to(device)
                else:
                    hist_batch = next(hist_iter).to(device)
                    sc_batch = next(sc_iter).to(device)
            except StopIteration:
                break
                
            batch_size = min(hist_batch.size(0), sc_batch.size(0))
            hist_batch = hist_batch[:batch_size]
            sc_batch = sc_batch[:batch_size]
            
            # Train discriminator
            disc_optimizer.zero_grad()
            
            fake_hist = generator(sc_batch)
            disc_real = discriminator(hist_batch)
            disc_fake = discriminator(fake_hist.detach())
            
            disc_loss = F.binary_cross_entropy_with_logits(
                disc_real, torch.ones_like(disc_real)
            ) + F.binary_cross_entropy_with_logits(
                disc_fake, torch.zeros_like(disc_fake)
            )
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train generator
            gen_optimizer.zero_grad()
            
            fake_hist = generator(sc_batch)
            disc_fake = discriminator(fake_hist)
            
            gen_loss = F.binary_cross_entropy_with_logits(
                disc_fake, torch.ones_like(disc_fake)
            )
            
            if args['use_cell_types']:
                # Add cell type consistency loss
                fake_ct_logits = ct_classifier(fake_hist)
                ct_loss = F.cross_entropy(fake_ct_logits, sc_ct)
                gen_loss = gen_loss + ct_loss
                epoch_metrics['ct_loss'] += ct_loss.item()
            
            gen_loss.backward()
            gen_optimizer.step()
            
            # Train decoders
            hist_dec_optimizer.zero_grad()
            sc_dec_optimizer.zero_grad()
            
            hist_recon = hist_decoder(hist_batch)
            sc_recon = sc_decoder(sc_batch)
            
            hist_recon_loss = F.mse_loss(hist_recon, hist_batch)
            sc_recon_loss = F.mse_loss(sc_recon, sc_batch)
            
            hist_recon_loss.backward()
            sc_recon_loss.backward()
            
            hist_dec_optimizer.step()
            sc_dec_optimizer.step()
            
            # Update metrics
            epoch_metrics['gen_loss'] += gen_loss.item()
            epoch_metrics['disc_loss'] += disc_loss.item()
            epoch_metrics['hist_recon_loss'] += hist_recon_loss.item()
            epoch_metrics['sc_recon_loss'] += sc_recon_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar_dict = {
                'gen_loss': gen_loss.item(),
                'disc_loss': disc_loss.item(),
                'hist_recon': hist_recon_loss.item(),
                'sc_recon': sc_recon_loss.item()
            }
            if args['use_cell_types']:
                pbar_dict['ct_loss'] = ct_loss.item()
            pbar.set_postfix(pbar_dict)
        
        # Calculate epoch averages
        for k, v in epoch_metrics.items():
            avg_metric = v / num_batches
            metrics_history[f'{k}s'].append(avg_metric)
            
            if args['use_wandb']:
                wandb.log({k: avg_metric, 'epoch': epoch})
        
        print(f'Epoch {epoch+1} Metrics:')
        for k, v in epoch_metrics.items():
            print(f'{k}: {v/num_batches:.4f}')
    
    # Save best models
    best_models = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'hist_decoder': hist_decoder.state_dict(),
        'sc_decoder': sc_decoder.state_dict()
    }
    if args['use_cell_types']:
        best_models['ct_classifier'] = ct_classifier.state_dict()
    
    return metrics_history, best_models
def setup_wandb(args: dict) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        args (dict): Training arguments
    """
    if args['use_wandb']:
        wandb.init(
            project="schaf",
            config=args,
            name=f"{args['scenario']}_{args['fold']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

def setup_device(gpu_id: int) -> torch.device:
    """
    Set up and return the training device.
    
    Args:
        gpu_id (int): GPU device ID to use
        
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA is not available. Using CPU instead.")
    
    return device

def setup_paired_training(config: dict, args: dict, stage1_model: nn.Module = None) -> Any:
    """
    Set up models and optimizer for paired training.
    
    Args:
        config (dict): Scenario configuration
        args (dict): Training arguments
        stage1_model (nn.Module): Optional stage 1 model for stage 2 training
        
    Returns:
        tuple: Model and optimizer
    """
    # Determine number of genes for model architecture
    if stage1_model is not None:
        # For stage 2, use the same architecture as stage 1 model
        # Get num_genes from the final layer's output dimension in part_two
        num_genes = None
        
        # Find the last Linear layer in part_two (this has out_features = num_genes)
        if hasattr(stage1_model, 'part_two'):
            for module in reversed(list(stage1_model.part_two.modules())):
                if isinstance(module, nn.Linear):
                    num_genes = module.out_features
                    break
        
        if num_genes is None:
            raise ValueError("Could not determine number of genes from stage1_model")
    else:
        # For stage 1, get number of genes from sample fold file
        fold_dir = config['output_dir']
        sample_file = None
        for file in os.listdir(fold_dir):
            if file.endswith('st.h5ad'):
                sample_file = os.path.join(fold_dir, file)
                break
        
        if sample_file is None:
            raise ValueError(f"No sample data file found in {fold_dir} to determine number of genes")
        
        sample_data = sc.read_h5ad(sample_file)
        num_genes = sample_data.shape[1]
    
    # Initialize model with required parameters
    model = MerNet(
        num_genes=num_genes,
        model_name='xj_transformer',
        pretrained=True,
        num_hidden_layers=3,
        pretrain_hist=True,
        pretrain_st='NONE'
    )
    
    # If stage1 model exists, load encoder weights
    if stage1_model is not None:
        # Get encoder state dict from stage1 model
        stage1_encoder_dict = {k: v for k, v in stage1_model.state_dict().items() if 'part_one' in k}
        
        # Load encoder weights into new model
        model_dict = model.state_dict()
        model_dict.update(stage1_encoder_dict)
        model.load_state_dict(model_dict)
    
    # Load pre-trained weights if available
    if os.path.exists(config['model_dir']):
        pretrained_path = os.path.join(
            config['model_dir'],
            f"{config['model_name_prefix']}_{args['fold']}_pretrained.pt"
        )
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    return model, optimizer

def setup_unpaired_training(args: dict) -> Any:
    """
    Set up models and optimizers for unpaired training.
    
    Args:
        args (dict): Training arguments
        
    Returns:
        tuple:
            - dict: Dictionary of models
            - dict: Dictionary of optimizers
    """
    # Initialize models
    generator = HEGen()
    discriminator = Discriminator()
    hist_decoder = HEDecoder()
    sc_decoder = StandardDecoder(SC_EMBEDDING_DIM)
    
    # Initialize optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args['lr'])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args['lr'])
    hist_dec_optimizer = optim.Adam(hist_decoder.parameters(), lr=args['lr'])
    sc_dec_optimizer = optim.Adam(sc_decoder.parameters(), lr=args['lr'])
    
    models = {
        'generator': generator,
        'discriminator': discriminator,
        'hist_decoder': hist_decoder,
        'sc_decoder': sc_decoder
    }
    
    optimizers = {
        'generator': gen_optimizer,
        'discriminator': disc_optimizer,
        'hist_decoder': hist_dec_optimizer,
        'sc_decoder': sc_dec_optimizer
    }
    
    return models, optimizers

def parse_args() -> dict:
    """
    Parse command line arguments for SCHAF training.
    
    Returns:
        dict: Dictionary containing all parsed arguments with their values
    """
    parser = argparse.ArgumentParser(
        description='Train SCHAF (Single-Cell Histology Alignment Framework)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    scenario_choices = list(SCENARIO_CONFIGS.keys()) + ['custom']
    parser.add_argument('--scenario', type=str, required=True,
                      choices=scenario_choices,
                      help='Training scenario to use (or "custom" for user-defined data)')
    parser.add_argument('--fold', type=str, required=True,
                      help='Fold/key to use as test set')
    parser.add_argument('--gpu', type=int, required=True,
                      help='GPU device to use')
    
    # Custom scenario arguments
    parser.add_argument('--custom-data-dir', type=str,
                      help='Directory containing custom data (required for custom scenario)')
    parser.add_argument('--custom-he-image', type=str,
                      help='Filename of H&E image in custom data directory')
    parser.add_argument('--custom-sc-file', type=str,
                      help='Filename of single-cell data file (.h5ad)')
    parser.add_argument('--custom-st-file', type=str,
                      help='Filename of spatial transcriptomics data file (.h5ad, for paired scenarios)')
    parser.add_argument('--custom-paired', action='store_true',
                      help='Use paired training (requires spatial transcriptomics data)')
    parser.add_argument('--custom-scenario-name', type=str, default='custom',
                      help='Name for the custom scenario (used in output paths)')
    
    # Optional arguments with defaults
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--val_size', type=float, default=DEFAULT_VAL_SIZE,
                      help='Validation set size (fraction)')
    parser.add_argument('--test_size', type=float, default=DEFAULT_TEST_SIZE,
                      help='Test set size (fraction)')
    parser.add_argument('--tile_radius', type=int, default=DEFAULT_TILE_RADIUS,
                      help='Radius of image tiles in pixels')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                      help='Mode: train or inference')
    
    # Flag arguments
    parser.add_argument('--use_wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--save_model', action='store_true',
                      help='Save model checkpoints during training')
    parser.add_argument('--use_cell_types', action='store_true',
                      help='Use cell type information in unpaired training loss')
    
    return vars(parser.parse_args())

#######################################################################################
# Inference Functions
#######################################################################################

def load_and_transform_data(config: Dict, 
                          fold: Optional[int] = None) -> Tuple[np.ndarray, sc.AnnData]:
    """Load and optionally transform the data based on configuration"""
    # Load H&E image
    he_path = os.path.join(config['data_dir'], config['he_path'])
    he_image = iio.imread(he_path)
    
    # Load projection data
    if fold is not None:
        proj_path = os.path.join(config['proj_dir'], f'fold_{fold}_tang_proj.h5ad')
        adata = sc.read_h5ad(proj_path)
    else:
        # Handle special cases (e.g., cancer_whole_sample)
        proj_path = os.path.join(config['proj_dir'], f'fold_0_tang_proj.h5ad')
        train_trans = sc.read_h5ad(proj_path)
        final_var = train_trans.var
        
        # Load and transform coordinates if needed
        if config['needs_transform']:
            cells_info = pd.read_csv(os.path.join(config['data_dir'], 'cells.csv'))
            cells_info = cells_info.set_index('cell_id')
            
            transform_df = pd.read_csv(
                os.path.join(config['data_dir'], config['transform_file']),
                header=None
            )
            transformation_matrix = transform_df.values.astype(np.float32)
            inv_trans = np.linalg.inv(transformation_matrix).astype('float64')
            
            verts = np.array(cells_info['y_centroid'])
            horis = np.array(cells_info['x_centroid'])
            xs, ys = transform_coordinates(horis, verts, inv_trans)
            
            final_obs = cells_info
            final_obs['x'] = xs
            final_obs['y'] = ys
            
            adata = sc.AnnData(
                X=np.zeros((verts.shape[0], train_trans.shape[1])),
                obs=final_obs,
                var=final_var,
            )
    
    return he_image, adata

def setup_model(config: Dict, 
                num_genes: int, 
                fold: Optional[int] = None, 
                device: str = 'cuda') -> nn.Module:
    """Set up and load the appropriate model based on configuration"""
    model = MerNet(
        num_genes=num_genes,
        model_name='xj_transformer',
        pretrained=True,
        num_hidden_layers=3,
        pretrain_hist=True,
        pretrain_st=os.path.join(
            config['model_dir'],
            f'fold_{fold}_final_model.pth'
            if fold is not None else
            f'{config["name"]}_stage2_whole_sample_final_model.pth'
        ),
        have_relu=False
    )
    
    # Load model weights
    weights_path = os.path.join(
        config['model_dir'],
        f'{config["name"]}_stage2_leave_out_fold_{fold}_final_model.pth'
        if fold is not None else
        f'{config["name"]}_stage2_whole_sample_final_model.pth'
    )
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    
    return model.to(device)

def run_inference(model: nn.Module, 
                 dataloader: DataLoader, 
                 device: str = 'cuda') -> np.ndarray:
    """Run inference using the provided model and dataloader"""
    pred_res = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for _id, (cur_batch) in tqdm(enumerate(dataloader)):
            [this_batch, this_label] = cur_batch
            this_batch = this_batch.to(device)
            predicted_labels = model(this_batch.float())
            pred_res.extend(predicted_labels.to('cpu').detach().numpy())
        torch.cuda.empty_cache()
    return np.array(pred_res)

def save_results(pred_res: np.ndarray, 
                adata: sc.AnnData, 
                config: Dict, 
                fold: Optional[int] = None):
    """Save inference results"""
    final_adata = sc.AnnData(X=pred_res, obs=adata.obs, var=adata.var)
    
    output_path = os.path.join(
        config['output_dir'],
        f'fold_{fold}.h5ad' if fold is not None else 'whole_sample.h5ad'
    )
    final_adata.write(output_path)

def compute_embedding_stats(embeddings: Dict[str, np.ndarray], 
                          key: str, 
                          embed_shape: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation for embeddings normalization"""
    all_means = np.zeros(embed_shape)
    all_vars = np.zeros(embed_shape)
    total_samples = 0
    
    # Compute means
    for k, v in embeddings.items():
        if k == key:
            continue
        all_means += v.shape[0] * v.mean(axis=0)
        total_samples += v.shape[0]
    all_means /= total_samples
    
    # Compute variances
    for k, v in embeddings.items():
        if k == key:
            continue
        all_vars += v.shape[0] * (v.var(axis=0) + (v.mean(axis=0) - all_means)**2)
    all_vars /= total_samples
    
    return all_means, np.sqrt(all_vars)

def load_sc_data(sc_dir: str, key: str) -> Dict[str, sc.AnnData]:
    """Load and preprocess single-cell data"""
    the_scs = {}
    for file in os.listdir(sc_dir):
        k = file.split('.')[0]
        if k == key:
            sc_adata = sc.read_h5ad(os.path.join(sc_dir, file))
            new_v = sc.AnnData(
                X=np.array(sc_adata.obsm['counts'].todense()),
                obs=sc_adata.obs
            )
            sc.pp.log1p(new_v)
            new_v.var.index = sc_adata.uns['counts_var']
            sc_adata = new_v
            sc_adata.var.index = [g.lower() for g in sc_adata.var.index]
            the_scs[k] = sc_adata
    
    return the_scs

def setup_htapp_model(config: Dict, 
                     sc_data: Dict[str, sc.AnnData], 
                     device: str = 'cuda',
                     key: str = None) -> nn.Module:
    """Set up and load the HTAPP model"""
    he_gen = HEGen()
    the_decoder = StandardDecoder(input_dim=sc_data[list(sc_data.keys())[0]].shape[1])
    model = TransferModel(he_gen, the_decoder)
    
    # Load model weights
    if key is not None:
        for file in os.listdir(config['model_dir']):
            if key in file:
                model.load_state_dict(
                    torch.load(os.path.join(config['model_dir'], file))['model_state_dict']
                )
                break
    
    return model.eval().to(device)

def run_image_based_inference(args: argparse.Namespace):
    """Run inference for image-based models (mouse, cancer_in_sample, cancer_whole_sample)"""
    config = SCENARIO_CONFIGS[args.dataset_type]
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load data
    he_image, adata = load_and_transform_data(config, args.fold)
    
    # Create dataloader
    dataset = ImageDataset(
        he_image,
        adata,
        config['tile_radius'],
        np.arange(adata.shape[0])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    
    # Setup model
    model = setup_model(config, adata.shape[1], args.fold, device)
    model.eval()
    
    # Run inference
    pred_res = run_inference(model, dataloader, device)
    
    # Save results
    save_results(pred_res, adata, config, args.fold)

def run_htapp_inference(args: argparse.Namespace):
    """Run inference for HTAPP data"""
    config = SCENARIO_CONFIGS['htapp']
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load embeddings
    htapp_hist_embeddings = {}
    for file in os.listdir(config['embeddings_dir']):
        k = file.split('.')[0]
        htapp_hist_embeddings[k] = np.load(os.path.join(config['embeddings_dir'], file))
    
    # Normalize embeddings
    hist_embed_mean, hist_embed_std = compute_embedding_stats(
        htapp_hist_embeddings,
        args.fold,
        config['hist_embed_shape']
    )
    
    for k, v in htapp_hist_embeddings.items():
        htapp_hist_embeddings[k] = (v - hist_embed_mean) / hist_embed_std
    
    # Create dataloader
    test_hist_dl = DataLoader(
        TensorDataset(torch.from_numpy(htapp_hist_embeddings[args.fold])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    
    # Load and process single-cell data
    sc_data = load_sc_data(config['sc_dir'], args.fold)
    
    # Setup and run model
    model = setup_htapp_model(config, sc_data, device)
    pred_res = run_inference(model, test_hist_dl, device)
    
    # Save results
    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(config['output_dir'], f'fold_{args.fold}.h5ad')
    
    # Create AnnData object with predictions
    final_adata = sc.AnnData(
        X=pred_res,
        obs=pd.DataFrame(index=range(len(pred_res))),
        var=sc_data[args.fold].var
    )
    final_adata.write(output_path)

def run_placenta_inference(args: argparse.Namespace):
    """Run inference for full placenta data"""
    config = SCENARIO_CONFIGS['placenta']
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load embeddings
    placenta_hist_embeddings = {}
    for file in os.listdir(config['embeddings_dir']):
        k = file.split('.')[0]
        placenta_hist_embeddings[k] = np.load(os.path.join(config['embeddings_dir'], file))
    
    # Load single-cell data
    the_sc = sc.read_h5ad(os.path.join(config['data_dir'], config['sc_file']))
    the_scs = {}
    
    # Process each sample
    for k, code in config['k_to_code'].items():
        the_scs[k] = the_sc[the_sc.obs['Sample'] == code].copy()
        the_scs[k].var['gene_col'] = list(the_scs[k].var.index)
        the_scs[k].X = np.array(the_scs[k].X.todense())
        sc.pp.log1p(the_scs[k])
        sc.pp.filter_cells(the_scs[k], min_genes=1)
        sc.pp.filter_genes(the_scs[k], min_cells=1)
    
    # Find common genes
    all_common_sc_var = the_scs[list(the_scs.keys())[0]].var.index
    for v in the_scs.values():
        all_common_sc_var = np.intersect1d(all_common_sc_var, v.var.index)
    
    # Filter to common genes
    for k, v in the_scs.items():
        the_scs[k] = v[:, all_common_sc_var].copy()
        sc.pp.filter_cells(the_scs[k], min_genes=1)
        sc.pp.filter_genes(the_scs[k], min_cells=1)
        the_scs[k].var['gene_col'] = list(the_scs[k].var.index)
    
    # Normalize embeddings
    hist_embed_mean, hist_embed_std = compute_embedding_stats(
        placenta_hist_embeddings,
        args.fold,
        config['hist_embed_shape']
    )
    
    for k, v in placenta_hist_embeddings.items():
        placenta_hist_embeddings[k] = (v - hist_embed_mean) / hist_embed_std
    
    # Create dataloader
    test_hist_dl = DataLoader(
        TensorDataset(torch.from_numpy(placenta_hist_embeddings[args.fold])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    
    # Setup model
    he_gen = HEGen()
    the_decoder = StandardDecoder(input_dim=the_scs[args.fold].shape[1])
    model = TransferModel(he_gen, the_decoder)
    
    # Load model weights
    for file in os.listdir(config['model_dir']):
        if args.fold in file:
            model.load_state_dict(
                torch.load(os.path.join(config['model_dir'], file))['model_state_dict']
            )
            break
    
    model = model.eval().to(device)
    
    # Run inference
    pred_res = run_inference(model, test_hist_dl, device)
    
    # Save results
    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(config['output_dir'], f'fold_{args.fold}.h5ad')
    
    final_adata = sc.AnnData(
        X=pred_res,
        obs=pd.DataFrame(index=range(len(pred_res))),
        var=the_scs[args.fold].var
    )
    final_adata.write(output_path)

def run_lung_cancer_inference(args: argparse.Namespace):
    """Run inference for lung cancer data"""
    config = SCENARIO_CONFIGS['lung_cancer']
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load embeddings
    lung_cancer_hist_embeddings = {}
    for file in os.listdir(config['embeddings_dir']):
        lung_cancer_hist_embeddings[file] = np.load(
            os.path.join(config['embeddings_dir'], file)
        )
    
    # Load SC embeddings
    lung_cancer_sc_embeddings = {}
    for file in os.listdir(config['sc_embeddings_dir']):
        k = file.split('.')[0]
        lung_cancer_sc_embeddings[k] = sc.read_h5ad(
            os.path.join(config['sc_embeddings_dir'], file)
        )
    
    # Consolidate embeddings
    new_lung_cancer_hist_embeddings = {}
    for k in lung_cancer_sc_embeddings:
        embs_list = []
        for k2, v in lung_cancer_hist_embeddings.items():
            if k in k2:
                embs_list.extend(v)
        new_lung_cancer_hist_embeddings[k] = np.array(embs_list)
    lung_cancer_hist_embeddings = new_lung_cancer_hist_embeddings
    
    # Load and process single-cell data
    adatas = {}
    for file in os.listdir(config['data_dir']):
        k = file.split('.')[0]
        adatas[k] = sc.read_h5ad(os.path.join(config['data_dir'], file))
        
        adatas[k].var['gene_col'] = list(adatas[k].var.index)
        adatas[k].X = adatas[k].layers['counts']
        sc.pp.log1p(adatas[k])
        sc.pp.filter_cells(adatas[k], min_genes=1)
        sc.pp.filter_genes(adatas[k], min_cells=1)
    
    # Normalize embeddings
    hist_embed_mean, hist_embed_std = compute_embedding_stats(
        lung_cancer_hist_embeddings,
        args.fold,
        config['hist_embed_shape']
    )
    
    for k, v in lung_cancer_hist_embeddings.items():
        lung_cancer_hist_embeddings[k] = (v - hist_embed_mean) / hist_embed_std
    
    # Create dataloader
    test_hist_dl = DataLoader(
        TensorDataset(torch.from_numpy(lung_cancer_hist_embeddings[args.fold])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    
    # Setup model
    he_gen = HEGen()
    the_decoder = StandardDecoder(input_dim=adatas[args.fold].shape[1])
    model = TransferModel(he_gen, the_decoder)
    
    # Load model weights
    for file in os.listdir(config['model_dir']):
        if args.fold in file:
            model.load_state_dict(
                torch.load(os.path.join(config['model_dir'], file))['model_state_dict']
            )
            break
    
    model = model.eval().to(device)
    
    # Run inference
    pred_res = run_inference(model, test_hist_dl, device)
    
    # Save results
    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(config['output_dir'], f'fold_{args.fold}.h5ad')
    
    final_adata = sc.AnnData(
        X=pred_res,
        obs=pd.DataFrame(index=range(len(pred_res))),
        var=adatas[args.fold].var
    )
    final_adata.write(output_path)

def load_embeddings(data_type: str, config: Dict) -> Dict[str, np.ndarray]:
    """Use generate_embeddings instead of loading from disk."""
    # Instead of loading, call generate_embeddings and return the result
    # This function will now call generate_embeddings and return the generated embeddings
    # For compatibility, we return an empty dict (actual usage will be via generate_embeddings)
    generate_embeddings(config['scenario'], config)
    return {}

def normalize_embeddings(hist_embeddings: Dict[str, np.ndarray], 
                       sc_embeddings: Dict[str, np.ndarray], 
                       key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize embeddings for unpaired training"""
    # Compute statistics for histology embeddings
    hist_embed_mean = np.zeros(hist_embeddings[list(hist_embeddings.keys())[0]].shape[1])
    hist_embed_std = np.zeros_like(hist_embed_mean)
    total_hist_samples = 0
    
    for k, v in hist_embeddings.items():
        if k == key:
            continue
        hist_embed_mean += v.shape[0] * v.mean(axis=0)
        total_hist_samples += v.shape[0]
    hist_embed_mean /= total_hist_samples
    
    for k, v in hist_embeddings.items():
        if k == key:
            continue
        hist_embed_std += v.shape[0] * (v.var(axis=0) + (v.mean(axis=0) - hist_embed_mean)**2)
    hist_embed_std = np.sqrt(hist_embed_std / total_hist_samples)
    
    # Compute statistics for single-cell embeddings
    sc_embed_mean = np.zeros(sc_embeddings[list(sc_embeddings.keys())[0]].shape[1])
    sc_embed_std = np.zeros_like(sc_embed_mean)
    total_sc_samples = 0
    
    for k, v in sc_embeddings.items():
        if k == key:
            continue
        sc_embed_mean += v.shape[0] * v.mean(axis=0)
        total_sc_samples += v.shape[0]
    sc_embed_mean /= total_sc_samples
    
    for k, v in sc_embeddings.items():
        if k == key:
            continue
        sc_embed_std += v.shape[0] * (v.var(axis=0) + (v.mean(axis=0) - sc_embed_mean)**2)
    sc_embed_std = np.sqrt(sc_embed_std / total_sc_samples)
    
    # Normalize embeddings
    hist_embeddings[key] = (hist_embeddings[key] - hist_embed_mean) / hist_embed_std
    sc_embeddings[key] = (sc_embeddings[key] - sc_embed_mean) / sc_embed_std
    
    return hist_embeddings[key], sc_embeddings[key]

def run_tangram(dataset: str, random_split: bool = True, custom_config: dict = None) -> None:
    """Run Tangram alignment for a specific dataset (random split only)"""
    # Load data
    if dataset == 'mouse':
        the_sc, the_st = load_mouse_data()
    elif dataset == 'custom':
        if not custom_config:
            raise ValueError("custom_config is required for custom datasets")
        the_sc, the_st = load_custom_data(
            custom_config['data_dir'],
            custom_config['sc_file'],
            custom_config.get('st_file')
        )
    else:
        the_sc, the_st = load_cancer_data()
    
    # Preprocess data
    the_sc, the_st = preprocess_data(the_sc, the_st)
    
    # Only random split supported
    all_genes = list(the_st.var.index)
    random.shuffle(all_genes)
    split_idx = len(all_genes) // 2
    train_genes = set(all_genes[:split_idx])
    
    # Process chunks
    if dataset == 'mouse':
        process_multiple_chunks(
            the_sc, the_st, train_genes, dataset,
            MOUSE_CHUNKS_DIR, 'random',
            PREP_CONFIGS[dataset]['num_chunks']
        )
    elif dataset == 'custom':
        # For custom datasets, use a smaller number of chunks
        chunks_dir = custom_config.get('chunks_dir', CUSTOM_DATA_PATHS['custom_chunks'])
        os.makedirs(chunks_dir, exist_ok=True)
        process_multiple_chunks(
            the_sc, the_st, train_genes, dataset,
            chunks_dir, 'random', 4  # Default to 4 chunks for custom data
        )
    else:
        process_multiple_chunks(
            the_sc, the_st, train_genes, dataset,
            CANCER_CHUNKS_DIR, 'random',
            PREP_CONFIGS[dataset]['num_chunks']
        )
    
    # Project genes
    project_genes(dataset, 'random', custom_config)
    return train_genes

def load_mouse_data() -> Tuple[sc.AnnData, sc.AnnData]:
    """Load mouse data for Tangram alignment (match all_tangram_final.py logic)"""
    xen_dir = MOUSE_XEN_DIR
    # Load spatial data
    the_st = sc.read_10x_h5(os.path.join(xen_dir, 'cell_feature_matrix.h5'))
    # Load single-cell data
    the_sc = sc.read_h5ad(os.path.join(xen_dir, 'mouse_pup.h5ad'))
    the_sc = the_sc[the_sc.obs['day']=='P0']
    the_sc.var = the_sc.var.set_index('gene_short_name')
    celltype_metadata = pd.read_pickle(os.path.join(xen_dir, 'mouse_metadata.pkl'))
    new_index = [q[:-5] for q in the_sc.obs.index]
    the_sc.obs.index = new_index
    the_sc.obs = celltype_metadata.loc[the_sc.obs.index]
    the_sc = the_sc[the_sc.obs['embryo_sex']=='M']
    the_sc = the_sc[:,~the_sc.var.index.duplicated(keep='first')]
    return the_sc, the_st

def load_cancer_data() -> Tuple[sc.AnnData, sc.AnnData]:
    """Load cancer data for Tangram alignment (match all_tangram_final.py logic)"""
    xen_dir = CANCER_XEN_DIR
    the_st = sc.read_h5ad(os.path.join(xen_dir, 'xenium_breast.h5ad'))
    the_sc = sc.read_h5ad(os.path.join(xen_dir, 'xenium_single_cell.h5'))
    return the_sc, the_st

def load_custom_data(data_dir: str, sc_file: str, st_file: str = None) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Load custom user data for Tangram alignment.
    
    Args:
        data_dir (str): Directory containing the data files
        sc_file (str): Filename of single-cell data (.h5ad)
        st_file (str, optional): Filename of spatial transcriptomics data (.h5ad)
        
    Returns:
        Tuple[sc.AnnData, sc.AnnData]: Single-cell and spatial transcriptomics data
    """
    # Load single-cell data
    the_sc = sc.read_h5ad(os.path.join(data_dir, sc_file))
    
    # Load spatial transcriptomics data if provided
    if st_file:
        the_st = sc.read_h5ad(os.path.join(data_dir, st_file))
    else:
        # Create dummy spatial data for unpaired scenarios
        the_st = the_sc.copy()
        print("Warning: No spatial transcriptomics data provided. Using single-cell data as template.")
    
    return the_sc, the_st

def setup_custom_scenario(args: dict) -> dict:
    """
    Set up configuration for a custom user scenario.
    
    Args:
        args (dict): Parsed command line arguments
        
    Returns:
        dict: Custom scenario configuration
    """
    # Validate required arguments for custom scenario
    if not args.get('custom_data_dir'):
        raise ValueError("--custom-data-dir is required for custom scenarios")
    if not args.get('custom_he_image'):
        raise ValueError("--custom-he-image is required for custom scenarios")
    if not args.get('custom_sc_file'):
        raise ValueError("--custom-sc-file is required for custom scenarios")
    
    # Validate data format
    validation = validate_custom_data_format(
        args['custom_data_dir'],
        args['custom_he_image'],
        args['custom_sc_file'],
        args.get('custom_st_file')
    )
    
    print("Custom data validation results:")
    for message in validation['messages']:
        print(f"  {message}")
    
    if not validation['valid']:
        raise ValueError("Custom data validation failed. Please check the error messages above.")
    
    # Create custom scenario configuration
    config = create_custom_scenario_config(
        scenario_name=args.get('custom_scenario_name', 'custom'),
        data_dir=args['custom_data_dir'],
        he_image_filename=args['custom_he_image'],
        is_paired=args.get('custom_paired', False),
        tile_radius=args.get('tile_radius', DEFAULT_TILE_RADIUS)
    )
    
    # Add custom-specific settings
    config.update({
        'sc_file': args['custom_sc_file'],
        'st_file': args.get('custom_st_file'),
        'data_info': validation['data_info'],
        'num_chunks': 4,  # Default number of chunks for custom data
        'chunks_dir': os.path.join(config['proj_dir'], 'chunks')  # Directory for chunk files
    })
    
    return config

def preprocess_data(the_sc: sc.AnnData, the_st: sc.AnnData) -> Tuple[sc.AnnData, sc.AnnData]:
    """Preprocess data for Tangram alignment"""
    # Process single-cell data
    sc.pp.filter_cells(the_sc, min_genes=8)
    sc.pp.filter_genes(the_sc, min_cells=1)
    sc.pp.normalize_total(the_sc)
    sc.pp.log1p(the_sc)
    
    # Process spatial data
    sc.pp.filter_cells(the_st, min_genes=8)
    sc.pp.filter_genes(the_st, min_cells=1)
    sc.pp.normalize_total(the_st)
    sc.pp.log1p(the_st)
    
    # Find common genes
    common_genes = np.intersect1d(the_sc.var.index, the_st.var.index)
    the_sc = the_sc[:, common_genes]
    the_st = the_st[:, common_genes]
    
    return the_sc, the_st


def process_single_chunk(the_sc: sc.AnnData, 
                        the_st: sc.AnnData, 
                        train_genes: Set[str], 
                        dataset: str, 
                        chunks_dir: str, 
                        split_type: str) -> None:
    """Process a single chunk for Tangram alignment"""
    # Create output directory
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Prepare data for Tangram
    the_sc_train = the_sc[:, list(train_genes)]
    the_st_train = the_st[:, list(train_genes)]
    
    # Run Tangram
    ad_map = tg.map_cells_to_space(
        adata_sc=the_sc_train,
        adata_sp=the_st_train,
        device=DEVICE
    )
    
    # Save results
    output_path = os.path.join(chunks_dir, f'{dataset}_{split_type}_chunk_0.h5ad')
    ad_map.write(output_path)

def process_multiple_chunks(the_sc: sc.AnnData, 
                          the_st: sc.AnnData, 
                          train_genes: Set[str], 
                          dataset: str, 
                          chunks_dir: str, 
                          split_type: str, 
                          n_splits: int) -> None:
    """Process multiple chunks for Tangram alignment"""
    # Create output directory
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Prepare data for Tangram
    the_sc_train = the_sc[:, list(train_genes)]
    the_st_train = the_st[:, list(train_genes)]
    
    # Split spatial data into chunks
    chunk_size = CHUNK_SIZE
    n_cells = the_st_train.shape[0]
    n_chunks = min(n_splits, math.ceil(n_cells / chunk_size))
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_cells)
        
        # Get chunk of spatial data
        chunk_st = the_st_train[start_idx:end_idx].copy()
        
        # Run Tangram on chunk
        tg.pp_adatas(the_sc_train, chunk_st)
        ad_map = tg.map_cells_to_space(
            adata_sc=the_sc_train,
            adata_sp=chunk_st,
            device=DEVICE
        )
        
        # Save results
        output_path = os.path.join(chunks_dir, f'{dataset}_{split_type}_chunk_{i}.h5ad')
        ad_map.write(output_path)

def project_genes(dataset: str, gene_split: str, custom_config: dict = None) -> None:
    """Project genes after Tangram alignment"""
    # Load original data
    if dataset == 'mouse':
        the_sc, the_st = load_mouse_data()
        chunks_dir = MOUSE_CHUNKS_DIR
        n_chunks = PREP_CONFIGS[dataset]['num_chunks']
        output_dir = PREP_CONFIGS[dataset]['output_dir']
    elif dataset == 'custom':
        if not custom_config:
            raise ValueError("custom_config is required for custom datasets")
        the_sc, the_st = load_custom_data(
            custom_config['data_dir'],
            custom_config['sc_file'],
            custom_config.get('st_file')
        )
        chunks_dir = custom_config.get('chunks_dir', CUSTOM_DATA_PATHS['custom_chunks'])
        n_chunks = 4  # Default for custom data
        output_dir = custom_config['proj_dir']
    else:
        the_sc, the_st = load_cancer_data()
        chunks_dir = CANCER_CHUNKS_DIR
        n_chunks = 1
        output_dir = PREP_CONFIGS[dataset]['output_dir']
    
    # Load mapping results
    mappings = []
    for i in range(n_chunks):
        mapping_path = os.path.join(chunks_dir, f'{dataset}_{gene_split}_chunk_{i}.h5ad')
        if os.path.exists(mapping_path):
            mappings.append(tg.project_genes(sc.read_h5ad(mapping_path), the_sc))
    
    if not mappings:
        raise ValueError(f"No mapping files found in {chunks_dir}")
    
    # Combine mappings if necessary
    if len(mappings) > 1:
        combined_mapping = sc.concat(mappings, join='outer')
    else:
        combined_mapping = mappings[0]
    
    # Project genes
    projected = combined_mapping
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset}_{gene_split}_projection.h5ad')
    projected.write(output_path)

#######################################################################################
# Main Function
#######################################################################################

def main():
    """Main training and inference function."""
    # Parse arguments
    args = parse_args()
    
    # Set up training environment
    device = setup_device(args['gpu'])
    setup_wandb(args)
    
    # Get scenario configuration
    if args['scenario'] == 'custom':
        config = setup_custom_scenario(args)
        print(f"Using custom scenario: {config}")
    else:
        config = SCENARIO_CONFIGS[args['scenario']]
    
    if args.get('mode', 'train') == 'train':
        if config['is_paired']:
            # stage 1
            if args['scenario'] == 'custom':
                prepare_paired_data(args['scenario'], force_recompute=False, custom_config=config)
            else:
                prepare_paired_data(args['scenario'], force_recompute=False)
            he_image, fold_to_trans, mu, sigma = load_data_for_scenario(config, args)
            train_loader, val_loader = prepare_paired_dataloaders(
                fold_to_trans, args['fold'] if config['use_hold_out'] else None,
                he_image, args, mu, sigma
            )
            model, optimizer = setup_paired_training(config, args)
            model = model.to(device)
            criterion = nn.MSELoss()
            train_losses, val_losses = train_paired_model(
                model, train_loader, val_loader, optimizer, criterion, device, args
            )

            # stage 2
            he_image, fold_to_trans, mu, sigma = load_data_for_scenario(config, args, stage2=True)
            train_loader, val_loader = prepare_paired_dataloaders(
                fold_to_trans, args['fold'] if config['use_hold_out'] else None,
                he_image, args, mu, sigma
            )
            model, optimizer = setup_paired_training(config, args, stage1_model=model)
            print(model)
            print(fold_to_trans)
            model = model.to(device)
            criterion = nn.MSELoss()
            train_losses, val_losses = train_paired_model(
                model, train_loader, val_loader, optimizer, criterion, device, args
            )
            if args['save_model']:
                final_path = os.path.join(
                    config['model_dir'],
                    f"{config['model_name_prefix']}_{args['fold']}_final.pt"
                )
                torch.save(model.state_dict(), final_path)
        else:
            # For unpaired scenarios, use generate_embeddings
            generate_embeddings(args['scenario'], config)
            hist_embeddings, sc_embeddings = load_data_for_scenario(config, args)
            hist_loader, sc_loader = prepare_unpaired_dataloaders(
                hist_embeddings, sc_embeddings, args
            )
            models, optimizers = setup_unpaired_training(args)
            metrics_history, best_models = train_unpaired_model(
                hist_loader, sc_loader, models, optimizers, device, args
            )
            if args['save_model']:
                for name, state_dict in best_models.items():
                    model_path = os.path.join(
                        config['save_models_dir'],
                        f"{name}_{args['fold']}_final.pt"
                    )
                    torch.save(state_dict, model_path)
    elif args.get('mode', 'train') == 'inference':
        # Inference logic (patterned after old_schaf_inference.py)
        if args['scenario'] in ['mouse', 'cancer_in_sample', 'cancer_whole_sample']:
            run_image_based_inference(args)
        elif args['scenario'] == 'htapp':
            run_htapp_inference(args)
        elif args['scenario'] == 'placenta':
            run_placenta_inference(args)
        elif args['scenario'] == 'lung_cancer':
            run_lung_cancer_inference(args)
        else:
            print(f"Inference for {args['scenario']} not yet implemented")

if __name__ == "__main__":
    main() 
