import os
import pandas as pd
import torch
import numpy as np
from scipy.spatial.transform import Rotation as rot
from torch.utils.data import Dataset
import torch.nn.functional as F
import trimesh
import math
import torch.nn as nn
import random
import matplotlib.pyplot as plt


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

class VoxelDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 data_root: str,
                 split: str = "train",
                 class_to_idx: dict = None,
                 num_classes: int = 10,
                 transform=None):
        """
        Args:
            csv_path:   Path to your metadata CSV.
            data_root:  Root folder containing the .torch files,
                        so that `os.path.join(data_root, object_path)` is valid.
            split:      "train" or "test"
            class_to_idx: Optional dict mapping class‐names to ints. 
                        If None, built from this split’s classes (sorted).
            num_classes: Number of classes utilized as labels
            transform:  Optional callable(sample) → sample
        """
        self.df = pd.read_csv(csv_path)
        # keep only the desired split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.num_classes = num_classes
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
        voxel_path = os.path.join(self.data_root, row["voxel_path"])
        voxel_sample = torch.load(voxel_path)

        label_idx = self.class_to_idx[row["class"]]
        label_onehot = torch.zeros(self.num_classes)
        label_onehot[label_idx] = 1.0  # one-hot encoding
        label_class = row["class"]

        sample = {
            "voxel":       voxel_sample,  # (G, G, G) occupancy grid
            "label_onehot": label_onehot,
            "label_class": label_class,
            "label": label_idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomRotate:
    """
    Rotate a binary occupancy grid by angle_degrees around a random axis between x, y and z.
    voxels: (D,H,W) tensor of 0/1 (uint8 or float)
    num_oreintation: How many different angle orientations can rotate it
    Augmentation: if true it rotates the 2 different voxels in voxel1 and voxel2, otherwise overwrites the one saved in voxel
    Returns: sample dictionary
    """
    def __init__(self, num_orientations: int = 4, num_classes: int = 10, augmentation = False):
        self.K = num_orientations
        # precompute angles
        self.angles = [2*torch.pi * i / self.K for i in range(self.K)]
        self.num_outputs = num_classes * self.K  # num_classes x num_orientations
        self.augmentation = augmentation

    def __call__(self, sample):
        if self.augmentation:
            return self.augmented_call(sample)
        else:
            return self.normal_call(sample)

    def normal_call(self, sample):
        # ensure float for grid_sample, add N,C dims
        grid = sample['voxel'][None, None]      # → (1,1,D,H,W)
        
        # pick a random orientation bin
        orient_idx = random.randrange(self.K)
        angle      = torch.tensor(self.angles[orient_idx], dtype=grid.dtype, device=grid.device)
        cos, sin = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([
                [ 1, 0,   0,    0],
                [ 0, cos, -sin, 0],
                [ 0, sin, cos,  0],
            ], dtype=grid.dtype, device=grid.device)
        theta = R[None]  # (1,3,4)
        # create normalized grid & sample with nearest‐neighbour
        x = F.affine_grid(theta, grid.shape, align_corners=False)   # (1,D,H,W,3)
        x_rot = F.grid_sample(grid, x,
                            mode='nearest',
                            padding_mode='zeros',
                            align_corners=False)
        

        # remove batch/channels and cast back to binary
        new_grid = (x_rot[0,0] > 0.5).to(grid.dtype)
        # write back rotated verts/faces
        sample['voxel'] = new_grid
        orientation = torch.zeros(self.num_outputs)
        idx_at_one = self.K*4 + orient_idx
        orientation[idx_at_one] = 1.0
        sample['orientation']   = orientation
        return sample

    def augmented_call(self, sample):
        # ensure float for grid_sample, add N,C dims
        grid = sample['voxel1'][None, None]      # → (1,1,D,H,W)
        
        # pick a random orientation bin
        orient_idx = random.randrange(self.K)
        angle      = torch.tensor(self.angles[orient_idx], dtype=grid.dtype, device=grid.device)
        axis = np.random.normal(size=3)
        axis = axis/np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
        rot_vec = angle*axis
        R = np.hstack((rot.from_rotvec(rot_vec).as_matrix(), np.zeros((3,1))))
        R = torch.tensor(R, dtype=grid.dtype, device=grid.device)
        theta = R[None]  # (1,3,4)
        # create normalized grid & sample with nearest‐neighbour
        x = F.affine_grid(theta, grid.shape, align_corners=False)   # (1,D,H,W,3)
        x_rot = F.grid_sample(grid, x,
                            mode='nearest',
                            padding_mode='zeros',
                            align_corners=False)
        

        # remove batch/channels and cast back to binary
        new_grid = (x_rot[0,0] > 0.5).to(grid.dtype)
        # write back rotated verts/faces
        sample['voxel1'] = new_grid
        orientation = torch.zeros(self.num_outputs)
        idx_at_one = self.K*4 + orient_idx
        orientation[idx_at_one] = 1.0
        sample['orientation1']   = orientation
        # ensure float for grid_sample, add N,C dims
        grid = sample['voxel2'][None, None]      # → (1,1,D,H,W)
        
        # pick a random orientation bin
        orient_idx = random.randrange(self.K)
        angle      = torch.tensor(self.angles[orient_idx], dtype=grid.dtype, device=grid.device)
        rot_axis = random.randrange(3)
        axis = np.zeros(3)
        axis[rot_axis] = 1
        rot_vec = angle*axis
        R = np.hstack((rot.from_rotvec(rot_vec).as_matrix(), np.zeros((3,1))))
        R = torch.tensor(R, dtype=grid.dtype, device=grid.device)
        theta = R[None]  # (1,3,4)
        # create normalized grid & sample with nearest‐neighbour
        x = F.affine_grid(theta, grid.shape, align_corners=False)   # (1,D,H,W,3)
        x_rot = F.grid_sample(grid, x,
                            mode='nearest',
                            padding_mode='zeros',
                            align_corners=False)
        

        # remove batch/channels and cast back to binary
        new_grid = (x_rot[0,0] > 0.5).to(grid.dtype)
        # write back rotated verts/faces
        sample['voxel2'] = new_grid
        orientation = torch.zeros(self.num_outputs)
        idx_at_one = self.K*4 + orient_idx
        orientation[idx_at_one] = 1.0
        sample['orientation2']   = orientation
        return sample

class RandomCropResize3D:
    """
    Given a sample dict with key 'voxel' of shape (D,H,W),
    extract a random (N x N x N) crop and resize it back to (D,H,W).
    Replaces sample['voxel'] with the cropped+resized grid if augmentation is false, 
    otherwise saves two copies of the same cropped voxel in voxel1 and voxel2.
    """
    def __init__(self, crop_size: int, output_size: int = 28, augmentation = False, max_tries=5):
        self.crop_size   = crop_size
        self.output_size = output_size
        self.augmentation = augmentation
        self.max_tries = max_tries
        assert crop_size <= output_size, "crop_size must be <= output_size"
    
    def __call__(self, sample):
        if self.augmentation:
            return self.augmented_call(sample)
        else:
            return self.normal_call(sample)

    def normal_call(self, sample):
        # original grid: [D,H,W]
        grid = sample['voxel']
        D, H, W = grid.shape
        assert (D, H, W) == (self.output_size,)*3, \
            f"Expected voxel of shape {(self.output_size,)*3}, got {grid.shape}"
        
        # pick random corner
        crop = self.sample_crop(grid)
        
        # to float and add batch/channel dims for interpolate
        vol = crop.unsqueeze(0).unsqueeze(0).float()  # [1,1,N,N,N]
        
        # resize back to [1,1,28,28,28]
        vol_resized = F.interpolate(
            vol,
            size=(self.output_size,)*3,
            mode='nearest',
            align_corners=None
        )
        
        # squeeze back and cast to original dtype (0/1 occupancy)
        new_grid = (vol_resized[0,0] > 0.5).to(grid.dtype)  # [28,28,28]
        sample['voxel'] = new_grid
        return sample
    
    def augmented_call(self, sample):
        # original grid: [D,H,W]
        grid = sample['voxel']
        
        # pick random corner
        crop = self.sample_crop(grid)
        
        # to float and add batch/channel dims for interpolate
        vol = crop.unsqueeze(0).unsqueeze(0).float()  # [1,1,N,N,N]
        
        # resize back to [1,1,28,28,28]
        vol_resized = F.interpolate(
            vol,
            size=(self.output_size,)*3,
            mode='nearest',
            align_corners=None
        )
        
        # squeeze back and cast to original dtype (0/1 occupancy)
        new_grid = (vol_resized[0,0] > 0.5).to(grid.dtype)  # [28,28,28]
        sample['voxel1'] = new_grid
        """# pick random corner
        crop = self.sample_crop(grid)
        
        # to float and add batch/channel dims for interpolate
        vol = crop.unsqueeze(0).unsqueeze(0).float()  # [1,1,N,N,N]
        
        # resize back to [1,1,28,28,28]
        vol_resized = F.interpolate(
            vol,
            size=(self.output_size,)*3,
            mode='nearest',
            align_corners=None
        )
        
        # squeeze back and cast to original dtype (0/1 occupancy)
        new_grid = (vol_resized[0,0] > 0.5).to(grid.dtype)  # [28,28,28]"""
        sample['voxel2'] = new_grid
        return sample
    
    def sample_crop(self, grid):
        D, H, W = grid.shape
        # if volume is completely empty, bail immediately
        if grid.sum() == 0:
            z0 = random.randint(0, D - self.crop_size)
            y0 = random.randint(0, H - self.crop_size)
            x0 = random.randint(0, W - self.crop_size)
            crop = grid[z0:z0+self.crop_size,
                        y0:y0+self.crop_size,
                        x0:x0+self.crop_size]
            return crop
        
        # otherwise rejection‐sample until we see a 1
        for _ in range(self.max_tries):
            z0 = random.randint(0, D - self.crop_size)
            y0 = random.randint(0, H - self.crop_size)
            x0 = random.randint(0, W - self.crop_size)
            crop = grid[z0:z0+self.crop_size,
                        y0:y0+self.crop_size,
                        x0:x0+self.crop_size]
            if crop.any():    # at least one 1 in the patch
                return crop
        
        # fallback: if we hit max_tries without success, just return the last one
        return crop

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
      - Conv3d(1→32, kernel=3, stride=2) + LeakyReLU(0.1)
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
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        """self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm3d(256)"""
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # compute size after convs + pool
        c1 = math.floor((grid_size - 3) / 2) + 1           # after conv1
        c2 = c1 - 3 + 1                                   # after conv2
        """c3 = c2 - 3 + 1                          # after conv3
        c4 = c3 - 3 + 1                           # after conv4 """
        p  = math.floor(c2 / 2)                           # after pool
        flattened_dim = 64 * p * p * p

        # fully connected layers
        self.fc1        = nn.Linear(flattened_dim, 128)
        self.fc_class   = nn.Linear(128, num_classes)
        self.fc_orient  = nn.Linear(128, num_orientations*num_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.6)


        # activations
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu       = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, G, G, G) occupancy grid
        returns: (class_logits, orient_logits)
        """
        x = self.leaky_relu(self.bn1(self.dropout1(self.conv1(x))))
        x = self.leaky_relu(self.bn2(self.dropout2(self.conv2(x))))
        """x = self.leaky_relu(self.bn3(self.dropout3(self.conv3(x))))
        x = self.leaky_relu(self.bn4(self.dropout4(self.conv4(x))))"""
        x = self.pool(x)

        x = x.view(x.size(0), -1)      # flatten
        x = self.relu(self.dropout3(self.fc1(x)))

        class_logits  = self.fc_class(x)
        orient_logits = self.fc_orient(x)
        return class_logits, orient_logits

def visualize_grid(grid: torch.Tensor):
    """
    Visualize a 3D occupancy grid as a 2D slice.
    :param grid: A 3D tensor of shape (G, G, G).
    :return: None
    """
    # voxels is your (D,H,W) array of 0s and 1s
    x, y, z = np.where(grid.clone().detach().numpy() == 1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()