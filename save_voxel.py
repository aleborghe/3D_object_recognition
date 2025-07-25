from utils import ModelNetDataset
from utils import ToVoxelGrid
import pandas as pd
import torch
from tqdm import tqdm



csv_path  = "../ModelNet10/metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10"        # root folder for the folders in object_path
voxel_dir = "../ModelNet10/voxel_grids"
transformation = ToVoxelGrid(grid_size=28)
train_ds = ModelNetDataset(csv_path, data_root, split="train")
test_ds  = ModelNetDataset(csv_path, data_root,
                            split="test",
                            class_to_idx=train_ds.class_to_idx)
# 1 · Read the CSV
df = pd.read_csv(csv_path)

# 2 · Add / overwrite a column
#    – constant value for every row
df["voxel_path"] = "voxel_" + df["object_path"].str[:-3] + "torch"

# 3 · Save the updated file
df.to_csv("../ModelNet10/new_metadata_modelnet10.csv", index=False)

print("✓ Column added and file saved.")
for i in tqdm(range(len(train_ds)),desc=f"Train dataset", leave=False):
    sample = transformation(train_ds[i])
    torch.save(sample["voxel"], sample["mesh_path"][:14] + "voxel_" + sample["mesh_path"][14:-3] + "torch")
for i in tqdm(range(len(test_ds)),desc=f"Test dataset", leave=False):
    sample = transformation(test_ds[i])
    torch.save(sample["voxel"], sample["mesh_path"][:14] + "voxel_" + sample["mesh_path"][14:-3] + "torch")
#for i in range(len(test_ds)):
#    sample = train_ds