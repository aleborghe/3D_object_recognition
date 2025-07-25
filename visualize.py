from utils import VoxelDataset
from utils import RandomRotateZ
import utils

csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10" 
rotation = RandomRotateZ(num_orientations=4)
train_ds = VoxelDataset(csv_path, data_root, split="train", transform=rotation)
normal_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)
idx = 500
sample = train_ds[500]  # Print the first sample to verify the dataset works


sample = normal_ds[500]  # Print the first sample to verify the dataset works
print("Sample voxel :", sample["label_class"])
utils.visualize_grid(sample["voxel"])

sample = train_ds[500]  # Print the first sample to verify the dataset works
print("Sample voxel :", sample["label_class"])
print("Rotation :", sample["orientation"])
utils.visualize_grid(sample["voxel"])
sample = train_ds[500]  # Print the first sample to verify the dataset works

print("Sample voxel :", sample["label_class"])
print("Rotation :", sample["orientation"])
utils.visualize_grid(sample["voxel"])