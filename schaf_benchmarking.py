"""
SCHAF Benchmarking Module

This module provides functionality for training and evaluating SCHAF models on various benchmarking datasets.
It supports both paired and unpaired training/inference modes.

"""

import os
import sys
import json
import math
import random
import datetime
import argparse
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Union, Optional

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as U
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
import torchvision
from torchvision import transforms
from torch.autograd import Variable

# Data Processing
import scanpy as sc
import imageio.v3 as iio
from PIL import Image
import PIL
from tqdm import tqdm
from numba import njit, prange
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, gaussian_kde, zscore

# Configure system settings
PIL.Image.MAX_IMAGE_PIXELS = 4017126500
NUM_WORKERS_CPU = 6  # 6 is better but more memory
PIN_MEM = 1  # 1 is better but more memory

#######################
# Model Classes
#######################

class StandardEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(StandardEncoder, self).__init__()
        self.part1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            nn.Linear(hidden_dim, latent_dim)
        )
        self.latent_dim = latent_dim
    
    def forward(self, x):
        return self.part1(x)

class StandardDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
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
        return self.net(x)

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=3):
        super(ConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 14 x 14
            nn.ReLU(),
            nn.Flatten(),  # Flatten before linear layer
            nn.Linear(128 * 14 * 14, 128)  # Latent space of 128 dimensions
        )
        self.latent_dim = latent_dim
        
    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, lin_dim, in_channels=3):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(128, 128 * 14 * 14),  # Output: 128 x 14 x 14
            nn.ReLU(),
            nn.Unflatten(1, (128, 14, 14)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 28 x 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: 32 x 56 x 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # Output: 16 x 112 x 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # Output: 3 x 224 x 224
        )
        self.latent_dim = latent_dim
        
    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, encoder, decoder, is_vae=False, use_latent_norm=True):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_vae = is_vae
        self.latent_normalizer = nn.BatchNorm1d(self.encoder.latent_dim)
        self.use_latent_norm = use_latent_norm
        
    def get_latent(self, x):
        mean = self.encoder(x)    
        mean = self.latent_normalizer(mean)
        return mean
    
    def forward(self, x):
        mean = self.encoder(x)
        latent = mean
        latent = self.latent_normalizer(latent)
        recon_x = self.decoder(latent)
        return recon_x, latent

class Discriminator(nn.Module):
    def __init__(self, latent_dim, spectral=True):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            U.spectral_norm(nn.Linear(latent_dim, 1<<6)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<6, 1<<5)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<5, 1<<5)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<5, 1<<1)),
        )
        
    def forward(self, x):
        return self.net(x)

class CTDiscriminator(nn.Module):
    def __init__(self, encoder, latent_dim=1<<7, spectral=True, end_dim=2):
        super(CTDiscriminator, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.net = nn.Sequential(
            U.spectral_norm(nn.Linear(latent_dim, 1<<6)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<6, 1<<5)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(1<<5, end_dim)),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.net(x)

class imgDiscriminator(nn.Module):
    def __init__(self, input_dim, in_channels=3, final_dim=2):
        super(imgDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.part1 = nn.Sequential(
            U.spectral_norm(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            U.spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            U.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            U.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),  # Flatten before linear layer
            nn.Linear(128 * 14 * 14, 128),  # Latent space of 128 dimensions
            nn.Linear(128, final_dim)  # Latent space of 128 dimensions
        )

    def forward(self, x):
        return self.part1(x)

class omicsDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, final_dim=2):
        super(omicsDiscriminator, self).__init__()
        self.part1 = nn.Sequential(
            U.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            U.spectral_norm(nn.Linear(hidden_dim, final_dim))
        )
    
    def forward(self, x):
        return self.part1(x)

class Generator(nn.Module):
    def __init__(self, img_to_omics, img_dim, omics_dim):
        super(Generator, self).__init__()
        omics_encoder = StandardEncoder(omics_dim, 1<<7, hidden_dim=1<<10) 
        omics_decoder = StandardDecoder(omics_dim, 1<<7, hidden_dim=1<<10)
        image_encoder = ConvEncoder(img_dim, 128)
        image_decoder = ConvDecoder(img_dim, 128, 0)
        
        if img_to_omics:
            self.net = VAE(
                image_encoder,
                omics_decoder,
                is_vae=False,
                use_latent_norm=True,
            )
        else:
            self.net = VAE(
                omics_encoder,
                image_decoder,
                is_vae=False,
                use_latent_norm=True,
            )

    def forward(self, x):
        return self.net(x)

class imageAE(nn.Module):
    def __init__(self, img_dim):
        super(imageAE, self).__init__()
        image_encoder = ConvEncoder(img_dim, 128)
        image_decoder = ConvDecoder(img_dim, 128, 0)
        self.net = VAE(
            image_encoder,
            image_decoder,
            is_vae=False,
            use_latent_norm=True,
        )

    def forward(self, x):
        return self.net(x)

class omicsAE(nn.Module):
    def __init__(self, omics_dim):
        super(omicsAE, self).__init__()
        omics_encoder = StandardEncoder(omics_dim, 1<<7, hidden_dim=1<<10) 
        omics_decoder = StandardDecoder(omics_dim, 1<<7, hidden_dim=1<<10)
        self.net = VAE(
            omics_encoder,
            omics_decoder,
            is_vae=False,
            use_latent_norm=True,
        )

    def forward(self, x):
        return self.net(x)

class TransferModel(nn.Module):
    def __init__(self, part1, part2):
        super(TransferModel, self).__init__()
        self.part_one = part1
        self.part_two = part2
        
    def forward(self, x):
        return self.part_two(self.part_one(x))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class HEGen(nn.Module):
    """H&E Generator model"""
    def __init__(self, latent_dim=1<<9):
        super(HEGen, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, 1<<10),
            nn.BatchNorm1d(1<<10),
            nn.ReLU(),
            nn.Linear(1<<10, latent_dim),
        )
        self.latent_dim = latent_dim
    
    def forward(self, x):
        return self.net(x)

#######################
# Dataset Classes
#######################

class HistSampleDataset(Dataset):
    def __init__(self, he_image, xs, ys, tile_radius, cts=None):
        self.he_image = he_image
        self.xs = xs
        self.ys = ys
        self.cts = torch.from_numpy(cts) if cts is not None else None
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
        img = np.zeros((self.tile_radius*2, self.tile_radius*2, 3))
        to_be = self.he_image[
            max(0, y-self.tile_radius):max(0, y+self.tile_radius),
            max(0, x-self.tile_radius):max(0, x+self.tile_radius),
        ]

        img[:to_be.shape[0],:to_be.shape[1]] = to_be

        if img.max() > 1.:
            img = img / 255.  # if is not in int form, get into int form 

        img = self.trnsfrms_val(img)  # now this is a tensor too 

        if self.cts is not None:
            return img, self.cts[index]
        return img 

#######################
# Utility Functions
#######################

def this_distance(A, B):
    """Compute mean absolute distance between two tensors."""
    return torch.mean(torch.abs(A - B))

def get_individual_distance_loss(A_i, A_j, AB_i, AB_j):
    """Compute distance loss between pairs of samples in original and transformed space."""
    distance_in_A = this_distance(A_i, A_j)
    distance_in_AB = this_distance(AB_i, AB_j)
    return torch.abs(distance_in_A - distance_in_AB)

def criteria_distance(the_input, the_output):
    """Compute distance-preserving loss between input and output spaces."""
    As = torch.split(the_input, 1)
    ABs = torch.split(the_output, 1)

    loss_distance_A = 0.0
    num_pairs = 0
    min_length = min(len(As), len(ABs))

    for i in range(min_length - 1):
        for j in range(i + 1, min_length):
            num_pairs += 1
            loss_distance_A_ij = get_individual_distance_loss(
                As[i], As[j],
                ABs[i], ABs[j],
            )
            loss_distance_A += loss_distance_A_ij

    loss_distance_A = loss_distance_A / num_pairs
    return loss_distance_A

def trans_good(x):
    """Transform data to match distribution"""
    if isinstance(x, np.ndarray):
        x = x.copy()
    else:
        x = x.copy().A
    
    # Handle zeros
    x[x == 0] = np.min(x[x > 0]) / 2
    
    # Log transform
    x = np.log2(x)
    
    # Standardize
    x = (x - np.mean(x)) / np.std(x)
    
    return x

def modify_ct(ct, use_ss=True):
    """Modify cell type labels"""
    if use_ss:
        if ct == 'Stromal-1':
            return 'CAF'
        if ct == 'Stromal-2':
            return 'CAF'
        if ct == 'Stromal-3':
            return 'CAF'
        if ct == 'Myeloid-1':
            return 'Myeloid'
        if ct == 'Myeloid-2':
            return 'Myeloid'
        if ct == 'Myeloid-3':
            return 'Myeloid'
        if ct == 'Myeloid-4':
            return 'Myeloid'
        if ct == 'T-cell-1':
            return 'T-cell'
        if ct == 'T-cell-2':
            return 'T-cell'
        if ct == 'T-cell-3':
            return 'T-cell'
        if ct == 'T-cell-4':
            return 'T-cell'
        if ct == 'B-cell':
            return 'B-cell'
        if ct == 'Tumor-1':
            return 'Tumor'
        if ct == 'Tumor-2':
            return 'Tumor'
        if ct == 'Endothelial':
            return 'Endothelial'
    return ct

def stratified_sample(tp, tgt, cell_type_column='umap_cts', seed=42):
    """Perform stratified sampling"""
    np.random.seed(seed)
    
    # Get cell type counts
    ct_counts = tp.obs[cell_type_column].value_counts()
    min_ct_count = min(ct_counts)
    
    # Sample from each cell type
    sampled_indices = []
    for ct in ct_counts.index:
        ct_indices = tp.obs[tp.obs[cell_type_column] == ct].index
        sampled_indices.extend(
            np.random.choice(ct_indices, size=min(len(ct_indices), tgt), replace=False)
        )
    
    return tp[sampled_indices]

def local_coherence_knn(adata, n_neighbors=10, cluster_column='umap_cts'):
    """Calculate local coherence using KNN"""
    from sklearn.neighbors import NearestNeighbors
    
    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(adata.X)
    distances, indices = nbrs.kneighbors(adata.X)
    
    # Get cell types
    cell_types = adata.obs[cluster_column].values
    
    # Calculate coherence
    coherence_scores = []
    for i in range(len(adata)):
        neighbors = indices[i][1:]  # Exclude self
        neighbor_types = cell_types[neighbors]
        own_type = cell_types[i]
        coherence = np.mean(neighbor_types == own_type)
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)

def knn_graph_separation(adata, k=15, cluster_col='umap_cts'):
    """Calculate KNN graph separation score"""
    from sklearn.neighbors import NearestNeighbors
    
    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=k).fit(adata.X)
    distances, indices = nbrs.kneighbors(adata.X)
    
    # Get cell types
    cell_types = adata.obs[cluster_col].values
    unique_types = np.unique(cell_types)
    
    # Calculate separation scores
    separation_scores = []
    for ct in unique_types:
        ct_mask = cell_types == ct
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) == 0:
            continue
            
        # Get KNN for this cell type
        ct_knn = indices[ct_mask]
        
        # Calculate fraction of same-type neighbors
        same_type_frac = np.mean([
            np.mean(cell_types[neighbors] == ct)
            for neighbors in ct_knn
        ])
        
        separation_scores.append(same_type_frac)
    
    return np.mean(separation_scores)

def calculate_pseudobulk_correlation(pred, tru, ct_label):
    """Calculate correlation between predicted and true pseudobulk profiles"""
    # Group by cell type
    pred_grouped = pred.groupby(ct_label).mean()
    tru_grouped = tru.groupby(ct_label).mean()
    
    # Calculate correlation
    common_cts = np.intersect1d(pred_grouped.index, tru_grouped.index)
    if len(common_cts) == 0:
        return 0
        
    pred_mat = pred_grouped.loc[common_cts]
    tru_mat = tru_grouped.loc[common_cts]
    
    # Calculate correlation for each gene
    correlations = []
    for gene in pred_mat.columns:
        if gene in tru_mat.columns:
            corr = np.corrcoef(pred_mat[gene], tru_mat[gene])[0,1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations)

def get_top_variable_genes(adata, n_top_genes=100):
    """Get top variable genes"""
    # Calculate variance for each gene
    gene_vars = np.var(adata.X, axis=0)
    
    # Get top genes
    top_indices = np.argsort(gene_vars)[-n_top_genes:]
    return adata.var_names[top_indices]

def get_corr_matr(adata):
    """Calculate correlation matrix"""
    return np.corrcoef(adata.X.T)

def get_ct_classifier(orig_adata, celltype_name):
    """Get cell type classifier from original data"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(orig_adata.X, orig_adata.obs[celltype_name])
    
    # Get label mapping
    label_to_anno = {i: label for i, label in enumerate(clf.classes_)}
    
    return clf, label_to_anno

def do_anno(anno_classifier, label_to_anno, pred_adata):
    """Apply cell type annotations to predicted data"""
    preds = anno_classifier.predict(pred_adata.X)
    pred_adata.obs['predicted_ct'] = [label_to_anno[p] for p in preds]
    return pred_adata

#######################
# Training Functions
#######################

def train_cycle_gan(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, n_epoch=10):
    """Train a cycle GAN model for unpaired translation."""
    # Initialize models
    netG_A2B = Generator(img_to_omics=True, img_dim=224, omics_dim=SC_NUM_GENES)
    netG_B2A = Generator(img_to_omics=False, img_dim=224, omics_dim=SC_NUM_GENES)
    netD_A = imgDiscriminator(224)
    netD_B = omicsDiscriminator(SC_NUM_GENES)

    # Move models to device
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    # Loss functions
    criterion_GAN = lambda a, b: F.binary_cross_entropy_with_logits(a, b)
    criterion_cycle = torch.nn.L1Loss()

    # Optimizers
    cycle_lambda = 5.
    da_lr = 1e-3
    db_lr = 1e-3
    gen_lr = 1e-3
    
    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=gen_lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=db_lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=da_lr, betas=(0.5, 0.999))

    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Training loop
    for epoch in tqdm(range(1, n_epoch+1)):
        da_loss = db_loss = gab_loss = gba_loss = aba_loss = bab_loss = 0
        
        for _id, the_batch in enumerate(zip(train_sc_dl, train_hist_dl)):
            ([real_B], real_A) = the_batch
            real_A = real_A.float().to(device)
            real_B = real_B.float().to(device)
            
            # Normalize omics data
            real_B = ((real_B - real_B.mean()) / real_B.std()).float()

            # Train generators
            optimizer_G.zero_grad()
            fake_label, real_label = [1., 0.], [0., 1.]

            # GAN loss
            fake_B = netG_A2B(real_A)[0]
            pred_fake = netD_B(fake_B)
            target_real = torch.tensor([real_label] * pred_fake.shape[0]).to(device)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)[0]
            pred_fake = netD_A(fake_A)
            target_real = torch.tensor([real_label] * pred_fake.shape[0]).to(device)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)[0]
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

            recovered_B = netG_A2B(fake_A)[0]
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            # Total generator loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + cycle_lambda*(loss_cycle_ABA + loss_cycle_BAB)
            loss_G.backward()
            optimizer_G.step()

            # Train discriminator A
            optimizer_D_A.zero_grad()
            pred_real = netD_A(real_A)
            target_real = torch.tensor([real_label] * pred_real.shape[0]).to(device)
            loss_D_real = criterion_GAN(pred_real, target_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            target_fake = torch.tensor([fake_label] * pred_fake.shape[0]).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train discriminator B
            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            target_real = torch.tensor([real_label] * pred_real.shape[0]).to(device)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            target_fake = torch.tensor([fake_label] * pred_fake.shape[0]).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # Update loss tracking
            da_loss += loss_D_A.item()
            db_loss += loss_D_B.item()
            gab_loss += loss_GAN_A2B.item()
            gba_loss += loss_GAN_B2A.item()
            aba_loss += loss_cycle_ABA.item()
            bab_loss += loss_cycle_BAB.item()

    # Save final model
    torch.save({
        'epoch': n_epoch,
        'model_state_dict': netG_A2B.state_dict(),
    }, save_path)

    return netG_A2B

def train_distance_gan(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, n_epoch=10):
    """Train a distance GAN model for unpaired translation."""
    # Initialize models
    netG_A2B = Generator(img_to_omics=True, img_dim=224, omics_dim=SC_NUM_GENES)
    netD_B = omicsDiscriminator(SC_NUM_GENES)

    netG_A2B.to(device)
    netD_B.to(device)

    # Loss functions
    criterion_GAN = lambda a, b: F.binary_cross_entropy_with_logits(a, b)

    # Optimizers
    distance_lambda = 5.
    db_lr = 1e-3
    gen_lr = 1e-3
    
    optimizer_G = torch.optim.Adam(netG_A2B.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=db_lr, betas=(0.5, 0.999))

    # Buffer
    fake_B_buffer = ReplayBuffer()

    # Training loop
    for epoch in tqdm(range(1, n_epoch+1)):
        db_loss = gab_loss = distance_loss = 0
        
        for _id, the_batch in enumerate(zip(train_sc_dl, train_hist_dl)):
            ([real_B], real_A) = the_batch
            real_A = real_A.float().to(device)
            real_B = real_B.float().to(device)
            
            # Normalize omics data
            real_B = ((real_B - real_B.mean()) / real_B.std()).float()

            # Train generator
            optimizer_G.zero_grad()
            fake_label, real_label = [1., 0.], [0., 1.]

            # GAN loss
            fake_B = netG_A2B(real_A)[0]
            pred_fake = netD_B(fake_B)
            target_real = torch.tensor([real_label] * pred_fake.shape[0]).to(device)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            # Distance loss
            loss_distance = criteria_distance(real_A, fake_B)

            # Total generator loss
            loss_G = loss_GAN_A2B + distance_lambda*loss_distance
            loss_G.backward()
            optimizer_G.step()

            # Train discriminator B
            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            target_real = torch.tensor([real_label] * pred_real.shape[0]).to(device)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            target_fake = torch.tensor([fake_label] * pred_fake.shape[0]).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # Update loss tracking
            db_loss += loss_D_B.item()
            gab_loss += loss_GAN_A2B.item()
            distance_loss += loss_distance.item()

    # Save final model
    torch.save({
        'epoch': n_epoch,
        'model_state_dict': netG_A2B.state_dict(),
    }, save_path)

    return netG_A2B

def train_caroline(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, use_cell_types=False, n_epoch=10):
    """Train Caroline's model for unpaired translation."""
    # Initialize models
    omics_model = omicsAE(SC_NUM_GENES)
    he_gen = imageAE(224)
    discrim = Discriminator(128)

    # Move models to device
    omics_model.to(device)
    he_gen.to(device)
    discrim.to(device)

    # Train omics autoencoder first
    criter = nn.MSELoss()
    lr = 1e-4
    optimizer = optim.AdamW(omics_model.parameters(), lr=lr)
    
    for epoch in tqdm(range(15)):  # 15 epochs for omics AE
        epoch_loss = 0.0
        omics_model.train()

        for [trans] in train_sc_dl:
            trans = trans.to(device)
            optimizer.zero_grad()
            trans = (trans - trans.mean()) / trans.std()
            predicted = omics_model(trans.float())[0]
            
            loss = criter(predicted, trans.float())
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

    # Train cell type classifier if needed
    if use_cell_types:
        NUM_CTS = 5
        ct_discrim = CTDiscriminator(omics_model.net.encoder, end_dim=NUM_CTS)
        ct_discrim.to(device)
        
        # Compute class weights
        num_cells = float(sum(len(batch[0]) for batch in train_sc_dl))
        cell_counts = defaultdict(int)
        for batch in train_sc_dl:
            _, cts = batch
            for ct in cts:
                cell_counts[ct.item()] += 1
        
        class_weights = [1. / cell_counts[k] for k in range(NUM_CTS)]
        class_weights = torch.tensor(class_weights).to(device)
        ct_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.AdamW(ct_discrim.parameters(), lr=1e-3)
        
        for epoch in range(15):  # 15 epochs for CT classifier
            epoch_loss = 0.0
            for [batch, labels] in train_sc_dl:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred_labels = ct_discrim(batch.float())
                loss = ct_criterion(pred_labels.double(), labels)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

    # Main training
    he_gen_lr = 1e-3
    discrim_lr = 4e-3
    beta = 1e-3
    beta2 = 5e0 if use_cell_types else 0

    he_gen_opt = optim.AdamW(he_gen.parameters(), lr=he_gen_lr)
    discrim_opt = optim.AdamW(discrim.parameters(), lr=discrim_lr)
    recon_criter = nn.MSELoss()

    for epoch in tqdm(range(1, n_epoch + 1)):
        he_gen.train()
        discrim.train()
        omics_model.eval()

        for _id, the_batch in enumerate(zip(train_sc_dl, train_hist_dl)):
            ([omics_batch, omics_cts], hist_batch) = the_batch

            he_gen_opt.zero_grad()
            discrim_opt.zero_grad()
            
            omics_batch = omics_batch.to(device)
            hist_batch = hist_batch.to(device)
            omics_batch = (omics_batch - omics_batch.mean()) / omics_batch.std()
            
            _, latent_batch = omics_model(omics_batch.float())
            he_decoded, hist_encoded = he_gen(hist_batch.float())
            
            hist_encoded = hist_encoded.detach()
            latent_batch = latent_batch.detach()

            # Train discriminator
            source_label, target_label = [1., 0.], [0., 1.]
            encodeds = torch.cat((latent_batch, hist_encoded), axis=0)
            discrim_labels = torch.tensor(
                [source_label] * latent_batch.shape[0] +
                [target_label] * hist_encoded.shape[0]
            ).to(device)

            pred_discrim_labels = discrim(encodeds.float())
            batch_discrim_loss = F.binary_cross_entropy_with_logits(
                pred_discrim_labels, discrim_labels,
            )

            batch_discrim_loss.backward()
            discrim_opt.step()

            # Train generator
            for param in discrim.parameters():
                param.requires_grad = False

            he_decoded, hist_encoded = he_gen(hist_batch.float())
            
            if use_cell_types:
                pred_hist_cts = ct_discrim.net(hist_encoded)
            
            hist_discrim_preds = discrim(hist_encoded)
            discrim_labels = torch.tensor([source_label] * hist_encoded.shape[0]).to(device)
            
            he_gen_loss = F.binary_cross_entropy_with_logits(
                hist_discrim_preds, discrim_labels,
            )

            he_recon_loss = recon_criter(
                hist_batch.float(),
                he_decoded,
            )

            if use_cell_types:
                he_celltype_loss = F.cross_entropy(
                    pred_hist_cts, hist_cts.float(),
                )
                together_loss = beta*he_gen_loss + he_recon_loss + beta2*he_celltype_loss
            else:
                together_loss = beta*he_gen_loss + he_recon_loss
            
            together_loss.backward()
            he_gen_opt.step()

            for param in discrim.parameters():
                param.requires_grad = True

    # Create and save final model
    tm = TransferModel(he_gen.net.encoder, omics_model.net.decoder).eval()
    tm = tm.to(device)

    torch.save({
        'epoch': n_epoch,
        'model_state_dict': tm.state_dict(),
    }, save_path)

    return tm 

def train_paired_mouse_benchmark(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, args, n_epoch=10):
    """Train paired mouse benchmark model"""
    the_net = get_benchmark_model(SC_NUM_GENES, args['benchmark'])
    the_net = the_net.to(device)
    the_net = the_net.train()

    # Use AdamW optimizer
    optimizer = optim.AdamW(the_net.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()
    best_loss = float("inf")

    # Handle deep_pt specific training if needed
    if args['benchmark'] == 'deep_pt':
        the_net.deep_pt_mid_trained = False
        for param in the_net.part_one_point_three.parameters():
            param.requires_grad = True

        for epoch in tqdm(range(1, 1 + args['num_epochs_deep_pt'])):
            epoch_loss = train_deep_pt_epoch(the_net, optimizer, criterion, train_hist_dl, train_sc_dl, device)
            val_loss = validate_model(the_net, criterion, val_data_loader, device, is_ae=True)
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(the_net, optimizer, epoch, save_path)

        the_net.deep_pt_mid_trained = True
        for param in the_net.part_one_point_three.parameters():
            param.requires_grad = False

    # Main training loop
    best_loss = float("inf")
    for epoch in tqdm(range(1, 1 + n_epoch)):
        epoch_loss = train_epoch(the_net, optimizer, criterion, train_hist_dl, train_sc_dl, device)
        val_loss = validate_model(the_net, criterion, val_data_loader, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(the_net, optimizer, epoch, save_path)

    return the_net

def train_paired_cancer_in_sample(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, args, n_epoch=10):
    """Train paired cancer in-sample benchmark model"""
    the_net = get_benchmark_model(SC_NUM_GENES, args['benchmark'])
    the_net = the_net.to(device)
    the_net = the_net.train()

    optimizer = optim.AdamW(the_net.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in tqdm(range(1, 1 + n_epoch)):
        epoch_loss = train_epoch(the_net, optimizer, criterion, train_hist_dl, train_sc_dl, device)
        val_loss = validate_model(the_net, criterion, val_data_loader, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(the_net, optimizer, epoch, save_path)

    return the_net

def train_paired_cancer_whole_sample(train_hist_dl, train_sc_dl, SC_NUM_GENES, device, save_path, args, n_epoch=10):
    """Train paired cancer whole-sample benchmark model"""
    the_net = get_benchmark_model(SC_NUM_GENES, args['benchmark'])
    the_net = the_net.to(device)
    the_net = the_net.train()

    optimizer = optim.AdamW(the_net.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in tqdm(range(1, 1 + n_epoch)):
        epoch_loss = train_epoch(the_net, optimizer, criterion, train_hist_dl, train_sc_dl, device)
        val_loss = validate_model(the_net, criterion, val_data_loader, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(the_net, optimizer, epoch, save_path)

    return the_net

def train_epoch(model, optimizer, criterion, train_hist_dl, train_sc_dl, device):
    """Generic training epoch function used by paired models"""
    epoch_loss = 0.0
    model.train()

    for batch_idx, (hist_batch, sc_batch) in enumerate(zip(train_hist_dl, train_sc_dl)):
        optimizer.zero_grad()
        
        hist_data = hist_batch[0].to(device)
        sc_data = sc_batch[1].to(device)
        
        pred = model(hist_data.float())
        loss = criterion(pred, sc_data.float())
        
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss

def train_deep_pt_epoch(model, optimizer, criterion, train_hist_dl, train_sc_dl, device):
    """Training epoch function specific to deep_pt model"""
    epoch_loss = 0.0
    model.train()

    for batch_idx, (hist_batch, sc_batch) in enumerate(zip(train_hist_dl, train_sc_dl)):
        optimizer.zero_grad()
        
        hist_data = hist_batch[0].to(device)
        sc_data = sc_batch[1].to(device)
        
        label, pred = model(hist_data.float())
        loss = criterion(pred, label)
        
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss

def validate_model(model, criterion, val_dl, device, is_ae=False):
    """Validation function for paired models"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (hist_batch, sc_batch) in enumerate(val_dl):
            hist_data = hist_batch[0].to(device)
            sc_data = sc_batch[1].to(device)
            
            if is_ae:
                label, pred = model(hist_data.float())
                loss = criterion(pred, label)
            else:
                pred = model(hist_data.float())
                loss = criterion(pred, sc_data.float())
                
            val_loss += loss.item()
    
    return val_loss

def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

#######################
# Inference Functions
#######################

def infer_unpaired(model, hist_dl, device, save_path, model_type='cycle'):
    """Run inference on unpaired model."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for hist_batch in tqdm(hist_dl):
            hist_batch = hist_batch.float().to(device)
            if model_type == 'cycle':
                pred = model(hist_batch)[0]
            else:  # caroline or distance
                pred = model(hist_batch)
            all_preds.append(pred.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    np.save(save_path, all_preds)
    return all_preds

def infer_paired(model, hist_dl, device, save_path):
    """Run inference on paired model."""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for hist_batch, sc_batch in tqdm(hist_dl):
            hist_batch = hist_batch.float().to(device)
            sc_batch = sc_batch.float().to(device)
            pred = model(hist_batch)
            all_preds.append(pred.cpu().numpy())
            all_true.append(sc_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    
    np.save(f"{save_path}_pred.npy", all_preds)
    np.save(f"{save_path}_true.npy", all_true)
    return all_preds, all_true

def infer_paired_mouse(model, test_dl, device, save_path):
    """Run inference for paired mouse benchmark model"""
    model.eval()
    pred_res = []
    
    with torch.no_grad():
        for batch_idx, (hist_batch, _) in enumerate(test_dl):
            hist_data = hist_batch[0].to(device)
            pred = model(hist_data.float())
            pred_res.extend(pred.cpu().detach().numpy())
    
    return np.array(pred_res)

def infer_paired_cancer_in_sample(model, test_dl, device, save_path):
    """Run inference for paired cancer in-sample benchmark model"""
    model.eval()
    pred_res = []
    
    with torch.no_grad():
        for batch_idx, (hist_batch, _) in enumerate(test_dl):
            hist_data = hist_batch[0].to(device)
            pred = model(hist_data.float())
            pred_res.extend(pred.cpu().detach().numpy())
    
    return np.array(pred_res)

def infer_paired_cancer_whole_sample(model, test_dl, device, save_path):
    """Run inference for paired cancer whole-sample benchmark model"""
    model.eval()
    pred_res = []
    
    with torch.no_grad():
        for batch_idx, (hist_batch, _) in enumerate(test_dl):
            hist_data = hist_batch[0].to(device)
            pred = model(hist_data.float())
            pred_res.extend(pred.cpu().detach().numpy())
    
    return np.array(pred_res)

def load_paired_model(model_path, model_type, num_genes, device):
    """Load a trained paired model"""
    model = get_benchmark_model(num_genes, model_type)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Handle deep_pt specific settings
    try:
        model.deep_pt_mid_trained = True
    except:
        pass
        
    return model

def evaluate_predictions(pred_data, true_data, cell_types=None):
    """Evaluate model predictions using various metrics."""
    results = {}
    
    # Gene-wise correlations
    gene_corrs = get_gene_corrs(pred_data, true_data)
    results['gene_correlations'] = {
        'mean': np.mean(gene_corrs),
        'median': np.median(gene_corrs),
        'std': np.std(gene_corrs)
    }
    
    # Cell-wise correlations
    cell_corrs = get_cell_corrs(pred_data, true_data)
    results['cell_correlations'] = {
        'mean': np.mean(cell_corrs),
        'median': np.median(cell_corrs),
        'std': np.std(cell_corrs)
    }
    
    # Pseudobulk correlation if cell types available
    if cell_types is not None:
        pb_corr = calculate_pseudobulk_correlation(pred_data, true_data, cell_types)
        results['pseudobulk_correlation'] = pb_corr
    
    return results

def run_benchmark(config):
    """Run a SCHAF benchmark experiment.
    
    Args:
        config: Dictionary containing benchmark configuration
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    date = str(datetime.date.today())
    
    # Determine benchmark type and mode
    benchmark_type = config.get('benchmark_type', 'unpaired')  # unpaired or paired
    benchmark_mode = config.get('benchmark_mode', 'cycle')     # cycle, distance, caroline for unpaired
                                                             # mouse, cancer_in_sample, cancer_whole_sample for paired
    
    # Setup paths
    if 'mounts' in os.getcwd().split("/ccomiter/")[0]:
        save_models_dir = '/mounts/stultzlab03_storage2/ccomiter/schaf_benchmarks/'
        infer_models_dir = '/mounts/stultzlab03_storage2/ccomiter/schaf_benchmarks_infer/'
    else:
        save_models_dir = '/storage2/ccomiter/schaf_benchmarks/'
        infer_models_dir = '/storage2/ccomiter/schaf_benchmarks_infer/'
    
    # Create model name
    name = '_'.join([
        'experiment',
        date,
        f'benchmark_{benchmark_type}_{benchmark_mode}',
        f'fold_{config.get("fold", 0)}'
    ])
    
    # Load and preprocess data
    train_hist_dl, train_sc_dl, val_dl, test_dl = load_benchmark_data(config)
    
    # Training
    if config.get('mode', 'train') == 'train':
        if benchmark_type == 'unpaired':
            if benchmark_mode == 'cycle':
                model = train_cycle_gan(train_hist_dl, train_sc_dl, config['num_genes'], 
                                      device, os.path.join(save_models_dir, f'{name}_final_model.pth'))
            elif benchmark_mode == 'distance':
                model = train_distance_gan(train_hist_dl, train_sc_dl, config['num_genes'],
                                         device, os.path.join(save_models_dir, f'{name}_final_model.pth'))
            else:  # caroline
                model = train_caroline(train_hist_dl, train_sc_dl, config['num_genes'],
                                     device, os.path.join(save_models_dir, f'{name}_final_model.pth'))
        else:  # paired
            if benchmark_mode == 'mouse':
                model = train_paired_mouse_benchmark(train_hist_dl, train_sc_dl, config['num_genes'],
                                                   device, os.path.join(save_models_dir, f'{name}_final_model.pth'),
                                                   config)
            elif benchmark_mode == 'cancer_in_sample':
                model = train_paired_cancer_in_sample(train_hist_dl, train_sc_dl, config['num_genes'],
                                                    device, os.path.join(save_models_dir, f'{name}_final_model.pth'),
                                                    config)
            else:  # cancer_whole_sample
                model = train_paired_cancer_whole_sample(train_hist_dl, train_sc_dl, config['num_genes'],
                                                       device, os.path.join(save_models_dir, f'{name}_final_model.pth'),
                                                       config)
    
    # Inference
    else:
        if benchmark_type == 'unpaired':
            model = load_model(config['model_path'], benchmark_mode, config['num_genes'], device)
            predictions = infer_unpaired(model, test_dl, device, 
                                       os.path.join(infer_models_dir, f'{name}_predictions.h5ad'),
                                       model_type=benchmark_mode)
        else:  # paired
            model = load_paired_model(config['model_path'], benchmark_mode, config['num_genes'], device)
            if benchmark_mode == 'mouse':
                predictions = infer_paired_mouse(model, test_dl, device,
                                              os.path.join(infer_models_dir, f'{name}_predictions.h5ad'))
            elif benchmark_mode == 'cancer_in_sample':
                predictions = infer_paired_cancer_in_sample(model, test_dl, device,
                                                         os.path.join(infer_models_dir, f'{name}_predictions.h5ad'))
            else:  # cancer_whole_sample
                predictions = infer_paired_cancer_whole_sample(model, test_dl, device,
                                                            os.path.join(infer_models_dir, f'{name}_predictions.h5ad'))
        
        # Evaluate predictions
        if config.get('evaluate', True):
            metrics = evaluate_predictions(predictions, test_dl.dataset.adata, 
                                        cell_types=test_dl.dataset.adata.obs.get('cell_type', None))
            
            # Save metrics
            with open(os.path.join(infer_models_dir, f'{name}_metrics.json'), 'w') as f:
                json.dump(metrics, f)
            
            return metrics
        
        return predictions

def main():
    """Main function to run benchmarking experiments."""
    parser = argparse.ArgumentParser(description='Run SCHAF benchmarking experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'],
                      help='Whether to run training or inference')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        results = run_benchmark(config)
        print("Training and evaluation complete. Results:")
        print(json.dumps(results, indent=4))
    else:
        # Load model
        model = torch.load(config['model_path'], map_location=device)
        model.eval()
        
        # Run inference
        if config['model_type'] in ['cycle', 'distance', 'caroline']:
            predictions = infer_unpaired(
                model, config['test_hist_dl'], device,
                config['save_path'],
                model_type=config['model_type']
            )
        else:
            predictions, true_data = infer_paired(
                model, config['test_hist_dl'], device,
                config['save_path']
            )
        print(f"Inference complete. Results saved to {config['save_path']}")

def load_benchmark_data(config):
    """Load and preprocess benchmark data.
    
    Args:
        config: Dictionary containing data configuration
        
    Returns:
        train_hist_dl: Training histology dataloader
        train_sc_dl: Training single-cell dataloader  
        val_dl: Validation dataloader
        test_dl: Test dataloader
    """
    # Load data based on benchmark type
    if config['benchmark_type'] == 'unpaired':
        # Load unpaired data
        if config['benchmark_mode'] == 'cycle':
            train_data = load_cycle_gan_data(config)
        elif config['benchmark_mode'] == 'distance':
            train_data = load_distance_gan_data(config)
        else:  # caroline
            train_data = load_caroline_data(config)
            
    else:  # paired
        if config['benchmark_mode'] == 'mouse':
            train_data = load_mouse_data(config)
        elif config['benchmark_mode'] == 'cancer_in_sample':
            train_data = load_cancer_in_sample_data(config)
        else:  # cancer_whole_sample
            train_data = load_cancer_whole_sample_data(config)
    
    # Create dataloaders
    train_hist_dl = DataLoader(
        train_data['train_hist'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS_CPU,
        pin_memory=PIN_MEM
    )
    
    train_sc_dl = DataLoader(
        train_data['train_sc'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS_CPU,
        pin_memory=PIN_MEM
    )
    
    val_dl = DataLoader(
        train_data['val'],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS_CPU,
        pin_memory=PIN_MEM
    )
    
    test_dl = DataLoader(
        train_data['test'],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS_CPU,
        pin_memory=PIN_MEM
    )
    
    return train_hist_dl, train_sc_dl, val_dl, test_dl

def load_mouse_data(config):
    """Load mouse benchmark data"""
    # Load H&E image
    xen_dir = f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/all_xenium_new_data/mouse_pup_data'
    he_image = iio.imread(os.path.join(xen_dir, 'Xenium_V1_mouse_pup_he_image.ome.tif'))
    
    # Load gene expression data
    fold_to_trans = {}
    for z in range(4):
        fold_to_trans[z] = sc.read_h5ad(
            f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/schaf_for_revision052424/data/xenium_cancer/mouse_folds/fold_{z}_{"st" if config["benchmark"] != "schaf_no_stage2" else "tang_proj"}.h5ad'
        )
        
        if config['benchmark'] != 'schaf_no_stage2':
            fold_to_trans[z] = fold_to_trans[z][::,config['train_genes']]
            sc.pp.log1p(fold_to_trans[z])
            fold_to_trans[z].X = np.array(fold_to_trans[z].X.todense())
    
    # Normalize data
    normalize_data(fold_to_trans, config['hold_out_fold'])
    
    # Create datasets
    datasets = create_datasets(fold_to_trans, he_image, config)
    
    return datasets

def load_cancer_in_sample_data(config):
    """Load cancer in-sample benchmark data"""
    # Load H&E image
    xen_dir = f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/htapp_supervise/new_schaf_experiment_scripts/more_data/xenium'
    he_image = iio.imread(os.path.join(xen_dir, 'xenium_hist.png'))
    
    # Load gene expression data
    fold_to_trans = {}
    for z in range(4):
        fold_to_trans[z] = sc.read_h5ad(
            f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/schaf_for_revision052424/data/xenium_cancer/cancer_in_sample_folds/fold_{z}_{"st" if config["benchmark"] != "schaf_no_stage2" else "tang_proj"}.h5ad'
        )
        
        if config['benchmark'] != 'schaf_no_stage2':
            fold_to_trans[z] = fold_to_trans[z][::,config['train_genes']]
            sc.pp.log1p(fold_to_trans[z])
            fold_to_trans[z].X = np.array(fold_to_trans[z].X.todense())
    
    # Normalize data
    normalize_data(fold_to_trans, config['hold_out_fold'])
    
    # Create datasets
    datasets = create_datasets(fold_to_trans, he_image, config)
    
    return datasets

def load_cancer_whole_sample_data(config):
    """Load cancer whole-sample benchmark data"""
    # Load H&E image
    xen_dir = f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/htapp_supervise/new_schaf_experiment_scripts/more_data/xenium'
    he_image = iio.imread(os.path.join(xen_dir, 'xenium_hist.png'))
    
    # Load gene expression data
    fold_to_trans = {}
    for z in range(4):
        fold_to_trans[z] = sc.read_h5ad(
            f'{os.getcwd().split("/ccomiter/")[0]}/ccomiter/schaf_for_revision052424/data/xenium_cancer/cancer_in_sample_folds/fold_{z}_{"st" if config["benchmark"] != "schaf_no_stage2" else "tang_proj"}.h5ad'
        )
        
        if config['benchmark'] != 'schaf_no_stage2':
            fold_to_trans[z] = fold_to_trans[z][::,config['train_genes']]
            sc.pp.log1p(fold_to_trans[z])
            fold_to_trans[z].X = np.array(fold_to_trans[z].X.todense())
    
    # Normalize data
    normalize_data(fold_to_trans)  # No hold out fold for whole sample
    
    # Create datasets
    datasets = create_datasets(fold_to_trans, he_image, config)
    
    return datasets

def normalize_data(fold_to_trans, hold_out_fold=None):
    """Normalize gene expression data"""
    num_genes = next(iter(fold_to_trans.values())).shape[1]
    
    # Calculate means
    all_means = np.zeros(num_genes)
    total_samples = 0
    for k, v in fold_to_trans.items():
        if hold_out_fold is not None and k == hold_out_fold:
            continue
        all_means += v.shape[0] * v.X.mean(axis=0)
        total_samples += v.shape[0]
    all_means /= total_samples
    
    # Calculate variances
    all_vars = np.zeros(num_genes)
    for k, v in fold_to_trans.items():
        if hold_out_fold is not None and k == hold_out_fold:
            continue
        all_vars += v.shape[0] * (v.X.var(axis=0) + (v.X.mean(axis=0) - all_means)**2)
    all_vars /= total_samples
    
    # Normalize data
    all_stds = np.sqrt(all_vars)
    for v in fold_to_trans.values():
        v.X = np.nan_to_num((v.X - all_means) / all_stds)

def create_datasets(fold_to_trans, he_image, config):
    """Create train/val/test datasets"""
    datasets = {
        'train_hist': [],
        'train_sc': [],
        'val': [],
        'test': []
    }
    
    for k, v in fold_to_trans.items():
        if config.get('hold_out_fold') is not None and k == config['hold_out_fold']:
            # Create test dataset
            test_ds = HistSampleDataset(
                he_image, v, config['tile_radius'],
                np.arange(v.shape[0])
            )
            datasets['test'] = test_ds
        else:
            # Split into train/val
            inds = np.arange(v.shape[0])
            if config['benchmark'] == 'low_data_schaf':
                num_total_sample = int(11111 / 3.)
                inds = random.sample(list(inds), num_total_sample)
                inds = np.array(inds)
                
            train_inds, val_inds = sklearn.model_selection.train_test_split(
                inds, test_size=config['val_size'], random_state=11
            )
            
            # Create train datasets
            train_ds = HistSampleDataset(
                he_image, v, config['tile_radius'],
                train_inds
            )
            datasets['train_hist'].append(train_ds)
            datasets['train_sc'].append(train_ds)
            
            # Create val dataset
            val_ds = HistSampleDataset(
                he_image, v, config['tile_radius'],
                val_inds
            )
            datasets['val'].append(val_ds)
    
    # Combine train/val datasets if multiple folds
    if len(datasets['train_hist']) > 1:
        datasets['train_hist'] = ConcatDataset(datasets['train_hist'])
        datasets['train_sc'] = ConcatDataset(datasets['train_sc'])
        datasets['val'] = ConcatDataset(datasets['val'])
    else:
        datasets['train_hist'] = datasets['train_hist'][0]
        datasets['train_sc'] = datasets['train_sc'][0]
        datasets['val'] = datasets['val'][0]
    
    return datasets

if __name__ == '__main__':
    main() 
