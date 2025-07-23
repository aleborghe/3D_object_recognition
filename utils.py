import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import trimesh
import math
import torch.nn as nn
import random


class ModelNetDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 data_root: str,
                 split: str = "train",
                 class_to_idx: dict = None,
                 transform=None):
        """
        Args:
            csv_path:   Path to your metadata CSV.
            data_root:  Root folder containing the .off files,
                        so that `os.path.join(data_root, object_path)` is valid.
            split:      "train" or "test"
            class_to_idx: Optional dict mapping class‐names to ints. 
                        If None, built from this split’s classes (sorted).
            transform:  Optional callable(sample) → sample
        """
        self.df = pd.read_csv(csv_path)
        # keep only the desired split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.data_root = data_root
        self.transform = transform

        # build (or reuse) class→index mapping
        if class_to_idx is None:
            classes = sorted(self.df["class"].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # full path to the .off file
        mesh_path = os.path.join(self.data_root, row["object_path"])
        mesh = trimesh.load_mesh(mesh_path)

        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        label = self.class_to_idx[row["class"]]
        label_class = row["class"]

        sample = {
            "vertices":    verts,
            "faces":       faces,
            "label":       label,
            "label_class": label_class,
            "mesh_path": mesh_path
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomRotateZ:
    """
    Rotate a mesh about the Z (up) axis by one of K equally spaced angles.
    Returns the rotated sample *and* an orientation label in [0..K-1].
    """
    def __init__(self, num_orientations: int = 4):
        self.K = num_orientations
        # precompute angles
        self.angles = [2*math.pi * i / self.K for i in range(self.K)]

    def __call__(self, sample):
        verts = sample['vertices'].numpy()
        faces = sample['faces'].numpy()
        mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # pick a random orientation bin
        orient_idx = random.randrange(self.K)
        angle      = self.angles[orient_idx]

        # build a Z‑rotation matrix about mesh centroid
        centroid = mesh.centroid
        T1 = trimesh.transformations.translation_matrix(-centroid)
        R  = trimesh.transformations.rotation_matrix(angle, [0,0,1])
        T2 = trimesh.transformations.translation_matrix(centroid)
        mesh.apply_transform(T2.dot(R).dot(T1))

        # write back rotated verts/faces
        sample['vertices'] = torch.from_numpy(mesh.vertices.astype(np.float32))
        sample['faces']    = torch.from_numpy(mesh.faces.astype(np.int64))
        sample['orientation']   = orient_idx
        return sample

class ToVoxelGrid:
    """
    Turn a mesh sample into a 28×28×28 occupancy grid
    """
    def __init__(self, grid_size: int = 28):
        self.grid_size = grid_size

    def __call__(self, sample):
        # 1. Rebuild trimesh and normalize to [0,1]^3
        verts = sample['vertices'].numpy()
        faces = sample['faces'].numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # translate so min bounds → 0
        min_b, max_b = mesh.bounds
        mesh.apply_translation(-min_b)
        # scale so max extent → 1
        scale = (max_b - min_b).max()
        mesh.apply_scale(1.0 / scale)

        # 2. Voxelize: pitch = 1 / grid_size
        pitch = 1.0 / self.grid_size
        vg = mesh.voxelized(pitch)

        # 3. Grab the raw matrix (boolean array)
        mat = vg.matrix  # shape = (nx, ny, nz), nx ≈ ny ≈ nz ≈ grid_size

        # 4. Pad or crop to exactly (G, G, G)
        G = self.grid_size
        grid = np.zeros((G, G, G), dtype=np.float32)
        # compute how many voxels on each axis
        nx, ny, nz = mat.shape
        cx, cy, cz = min(nx, G), min(ny, G), min(nz, G)
        grid[:cx, :cy, :cz] = mat[:cx, :cy, :cz]

        # 5. Store in sample
        sample['voxel'] = torch.from_numpy(grid)
        sample.pop('vertices', None)
        sample.pop('faces',    None)
        return sample

class ORIONNet(nn.Module):
    """
    Orientation‑boosted Voxel Net (ORION) for 3D object recognition.
    
    Architecture (VoxNet backbone):
      - Conv3d(1→32, kernel=5, stride=2) + LeakyReLU(0.1)
      - Conv3d(32→32, kernel=3, stride=1) + LeakyReLU(0.1)
      - MaxPool3d(kernel=2)
      - Flatten
      - FC(32 * p^3 → 128) + ReLU
      - Branch1: FC(128 → num_classes)        # object category
      - Branch2: FC(128 → num_orientations)   # orientation label
    where p = floor((floor((grid_size - 5)/2) + 1 - 3 + 1) / 2)
    """
    def __init__(self,
                 num_classes: int,
                 num_orientations: int,
                 grid_size: int = 28):
        super().__init__()
        self.num_classes      = num_classes
        self.num_orientations = num_orientations

        # 3D conv backbone
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1)
        self.pool  = nn.MaxPool3d(kernel_size=2)

        # compute size after convs + pool
        c1 = math.floor((grid_size - 5) / 2) + 1           # after conv1
        c2 = c1 - 3 + 1                                   # after conv2
        p  = math.floor(c2 / 2)                           # after pool
        flattened_dim = 32 * p * p * p

        # fully connected layers
        self.fc1        = nn.Linear(flattened_dim, 128)
        self.fc_class   = nn.Linear(128, num_classes)
        self.fc_orient  = nn.Linear(128, num_orientations)

        # activations
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu       = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, G, G, G) occupancy grid
        returns: (class_logits, orient_logits)
        """
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)      # flatten
        x = self.relu(self.fc1(x))

        class_logits  = self.fc_class(x)
        orient_logits = self.fc_orient(x)
        return class_logits, orient_logits