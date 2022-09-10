import torch
import torch.nn as nn
import torch.nn.functional as F

def radian2vec(x):
    a, b = torch.cos(x), torch.sin(x)
    return torch.concat((a, b), dim=1)

def vec2radian(x):
    return torch.atan2(x[:, 1:], x[:, :1])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # self.conv1 = nn.Conv3d(15, 64, (3, 3, 3), 2)
        # self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), 2)

        self.conv1 = nn.Conv3d(15, 32, (3, 3, 3), 2)
        self.conv2 = nn.Conv3d(32, 64, (3, 3, 3), 2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, (3, 3, 3), 2)
        
        self.scaleNet = nn.Linear(1024, 3)
        self.rotNet = nn.Linear(1024, 2)
        self.locNet = nn.Linear(1024, 3)

        self.dropout = nn.Dropout()

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        
        scale = self.scaleNet(x)
        rot = self.rotNet(x)
        loc = self.locNet(x)
        
        return scale, rot, loc
    
    def step(self, x, y):
        scale, rot, loc = self.forward(x)

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.cosine_similarity(rot, radian2vec(y[:, 3: 4])).mean()
        L3 = self.criterion(loc, y[:, 4:])

        rot = vec2radian(rot)
        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()

#         self.conv1 = nn.Conv3d(15, 64, (3, 3, 3), 2)
#         self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), 2)
#         self.L1 = nn.Linear(1024, 7)

#         self.criterion = torch.nn.MSELoss()
    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = torch.flatten(x, start_dim=1)
#         x = self.L1(x)
        
#         return x
    
#     def step(self, x, y):
#         x = self.forward(x)

#         L1 = self.criterion(x[:, :3], y[:, :3])
#         L2 = self.criterion(x[:, 3], y[:, 3])
#         L3 = self.criterion(x[:, 4:], y[:, 4:])

#         return (L1, L2, L3), x
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L1 = nn.Linear(19965, 512)
        self.L2 = nn.Linear(512, 7)

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = self.L2(x)
        return x
    
    def step(self, x, y):
        x = self.forward(x)

        L1 = self.criterion(x[:, :3], y[:, :3])
        L2 = self.criterion(x[:, 3], y[:, 3])
        L3 = self.criterion(x[:, 4:], y[:, 4:])

        return (L1, L2, L3), x