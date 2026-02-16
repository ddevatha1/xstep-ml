import torch.nn as nn
import torch.nn.functional as F

class UlcerNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3) # (3, 64, 64) -> (32, 62, 62)
        self.pool = nn.MaxPool2d(2, 2) # (32, 62, 62) -> (32, 31, 31)
        self.conv2 = nn.Conv2d(32, 64, 3) # (32, 31, 31) -> (64, 29, 29)
        # pooled: (64, 29, 29) -> (64, 14, 14)

        self.fc1 = nn.Linear(64 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = out.flatten(1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        out = self.out(out)
        return out   