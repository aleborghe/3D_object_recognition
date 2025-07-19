import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh

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
            "label_class": label_class
        }

        if self.transform:
            sample = self.transform(sample)

        return sample