from utils import ModelNetDataset
from utils import ToVoxelGrid
from utils import ORIONNet
from utils import RandomRotateZ
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import transforms

if __name__ =="__main__":
    csv_path  = "../ModelNet10/metadata_modelnet10.csv"  # or wherever you put it
    data_root = "../ModelNet10"        # root folder for the folders in object_path

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA device available")

    # build a single transform pipeline:
    augment = transforms.Compose([
        RandomRotateZ(num_orientations=4),
        ToVoxelGrid(grid_size=28)
    ])

    # Create train & test sets, sharing the same class→idx map:
    train_ds = ModelNetDataset(csv_path, data_root, split="train", transform=augment)
    test_ds  = ModelNetDataset(csv_path, data_root,
                            split="test",
                            class_to_idx=train_ds.class_to_idx, transform=augment)
    
    num_classes = len(train_ds.class_to_idx)
    print(f"Found {num_classes} classes: {train_ds.class_to_idx}")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                        num_workers=3,        # spawn 3 worker processes
                        pin_memory=True,      # helps when transferring to GPU
                        prefetch_factor=1)     # how many batches each worker preloads)

    # now each sample has `sample['voxel']` of shape [28,28,28]
    """for batch in loader:
        voxels = batch['voxel']       # torch.FloatTensor (B, 28, 28, 28)
        labels = batch['label_class']
        orientations = batch['orientation']
        print(voxels.shape)
        print(labels)
        print(orientations)"""
    model = ORIONNet(
        num_classes=num_classes,          # e.g. ModelNet10
        num_orientations=4,     # your chosen orientation‐bin count
        grid_size=28             # matches your voxel transform
    )
    model.to(device)

# in training loop:
    for batch in loader:
        voxels = batch['voxel'].unsqueeze(1).to(device)     # shape (B,1,28,28,28)
        cls_gt  = batch['label'].to(device)                 # shape (B,)
        ori_gt  = batch['orientation'].to(device) # shape (B,)

        cls_logits, ori_logits = model(voxels)
        print(cls_logits)
        print(cls_gt)
        loss = 0.5*F.cross_entropy(cls_logits, cls_gt) \
            + 0.5*F.cross_entropy(ori_logits, ori_gt)
        print(f"Class estimated {cls_logits}")
        print(f"Class real {cls_gt}")
        print(loss)