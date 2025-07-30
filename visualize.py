from utils import VoxelDataset, RandomRotate, RandomCropResize3D
from cl_utils import Encoder, HeadClassification
import random
import torch
import utils
from torchvision.transforms import Compose

csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"
data_root = "../ModelNet10" 
encoder_path = "CL_RESULTS/encoder_parameters.torch"
head_path = "FINAL_CLASSIFICATION_HEAD/net_parameters.torch"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

labels = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night', 'sofa', 'table', 'toilet']

encoder = Encoder()
encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()
transform_head = HeadClassification(input_dim=128)
transform_head.to(device)
transform_head.load_state_dict(torch.load(head_path))
transform_head.eval()

crop = RandomCropResize3D(21, augmentation=True)
rotation = RandomRotate(num_orientations=4, augmentation=True)
transform = Compose([crop, rotation])
train_ds = VoxelDataset(csv_path, data_root, split="train", transform=transform)
normal_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)
idx = random.choice(range(len(train_ds)))


sample = normal_ds[idx]  # Print the first sample to verify the dataset works
print("Sample voxel :", sample["label_class"])
pred_label = transform_head(encoder(sample["voxel"].unsqueeze(0).unsqueeze(0).to(device))).argmax().item()
print(f"Predicted label: {labels[pred_label]}")
utils.visualize_grid(sample["voxel"])

sample = train_ds[idx]  

utils.visualize_grid(sample["voxel1"])
utils.visualize_grid(sample["voxel2"])