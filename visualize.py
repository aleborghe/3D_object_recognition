from utils import VoxelDataset
from utils import RandomRotate
from utils import RandomCropResize3D
import utils
from torchvision.transforms import Compose

csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10" 
crop = RandomCropResize3D(21, augmentation=True)
rotation = RandomRotate(num_orientations=8, augmentation=True)
transform = Compose([crop, rotation])
train_ds = VoxelDataset(csv_path, data_root, split="train", transform=transform)
normal_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)
idx = 500
sample = train_ds[500]  # Print the first sample to verify the dataset works


sample = normal_ds[500]  # Print the first sample to verify the dataset works
print("Sample voxel :", sample["label_class"])
utils.visualize_grid(sample["voxel"])

sample = train_ds[500]  # Print the first sample to verify the dataset works
print("Sample voxel shape:", sample["voxel1"].shape)
#print("Rotation :", sample["orientation"])
utils.visualize_grid(sample["voxel1"])
#sample = train_ds[500]  # Print the first sample to verify the dataset works

print("Sample voxel shape:", sample["voxel2"].shape)
#print("Rotation :", sample["orientation"])
utils.visualize_grid(sample["voxel2"])