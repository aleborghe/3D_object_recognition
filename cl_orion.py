import torch
from torch import nn
import math

class Encoder(nn.Module):
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
    def __init__(self):
        super().__init__()


        # 3D conv backbone
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(8000, 128)
        #self.linear2 = nn.Linear(1024, 128)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)


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

        flat_x = x.view(x.size(0), -1)      # flatten
        flat_x = self.leaky_relu(self.dropout3(self.linear1(flat_x)))
        #representation_vec = self.leaky_relu(self.linear2(flat_x))

        return flat_x

class HeadContrastive(nn.Module):
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
    def __init__(self, input_dim=128, hidden_dim=1024, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class HeadClassification(nn.Module):
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
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)    

class NTXentLoss():
    def __init__(self, temperature=0.5):
        self.temperature = temperature

    def __call__(self, z_q, z_plus):
        """
        z_i, z_j: two batches of embeddings (augmented views), shape [batch, dim]
        """

        # 1) Normalize **both** views
        z_q    = nn.functional.normalize(z_q,    dim=1)  # [N, D]
        z_plus = nn.functional.normalize(z_plus, dim=1)  # [N, D]

        # 2) Build 2N batch
        z = torch.cat([z_q, z_plus], dim=0)   # [2N, D]
        N = z_q.shape[0]

        # 3) Cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # 4) Mask out self‑similarities
        mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -9e15)

        # 5) Positive‐pair similarities:
        #    for i in [0..N), the positive of i is i+N; and vice‑versa
        pos_idx = torch.arange(N, device=z.device)
        pos_sim = torch.cat([sim[pos_idx, pos_idx + N],
                             sim[pos_idx + N, pos_idx]], dim=0)                       # [2N]

        # 6) NT‐Xent loss
        exp_sim = torch.exp(sim)           # [2N,2N]
        denom   = exp_sim.sum(dim=1)       # sum over all negatives
        loss = -torch.log(torch.exp(pos_sim) / denom)
        return loss.mean()