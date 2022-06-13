import torch.nn as nn
import torch.nn.functional as F

class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=8)
        self.fc6 = nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x