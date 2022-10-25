import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms

# class SmallConv(nn.Module):
#     def __init__(self, configs):
#         super(SmallConv, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=configs['in_channels'], out_channels=6, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, configs['num_classes'])
            

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))

#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class SmallConv(nn.Module):
    def __init__(self, configs):
        super(SmallConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=configs['in_channels'], out_channels=32, kernel_size=5, stride=1, padding=2)
        #3*32*32 -> 32*32*32
        self.dropout1 = nn.Dropout(p=configs['dropout'])        
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #32*32*32 -> 16*16*32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        #16*16*32 -> 16*16*64
        self.dropout2 = nn.Dropout(p=configs['dropout'])
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #16*16*64 -> 8*8*64
        # self.fc1 = nn.Linear(8*8*64, 1024) # This line for CIFAR. TODO
        self.fc1 = nn.Linear(3136, 1024)
        self.dropout3 = nn.Dropout(p=configs['dropout'])
        self.fc2 = nn.Linear(1024, 512)
        self.dropout4 = nn.Dropout(p=configs['dropout'])
        self.fc3 = nn.Linear(512, configs['num_classes'])
            

    def forward(self, x):
        x = self.dropout1(self.conv1(x))
        x = self.pool1(F.relu(x))
        x = self.dropout2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
