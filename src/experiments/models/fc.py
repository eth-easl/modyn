import torch.nn as nn
import torch
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, config):
        super(FCNet, self).__init__()
        widths = config['widths']
        self.fc1 = nn.Linear(widths[0], widths[1])  
        self.fc2 = nn.Linear(widths[1], widths[2])

    def forward(self, x):
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
