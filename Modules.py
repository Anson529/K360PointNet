import torch
import torch.nn as nn
import torch.nn.functional as F

def radian2vec(x):
    a, b = torch.cos(x), torch.sin(x)
    return torch.concat((a, b), dim=1)

def vec2radian(x):
    return torch.atan2(x[:, 1:], x[:, :1])

class Predictor(nn.Module):
    def __init__(self, dim=128):
        super(Predictor, self).__init__()

        self.scaleNet = nn.Linear(dim, 3)
        # self.rotNet = nn.Linear(dim, 2)
        hidden = 128
        self.rotNet = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.locNet = nn.Linear(dim, 3)
    
    def forward(self, x):
        scale = self.scaleNet(x)
        rot = self.rotNet(x)
        loc = self.locNet(x)
        
        return scale, rot, loc

class ConvNet_2d(nn.Module):
    def __init__(self, feat=False):
        super(ConvNet_2d, self).__init__()
        self.feat = feat

        dim = 32
        
        self.conv1 = nn.Conv2d(36, dim, (3, 3), 2)
        self.conv2 = nn.Conv2d(dim, dim * 2, (3, 3), 2, padding=1)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, (3, 3), 2)

        self.scaleNet = nn.Linear(dim * 16, 3)
        self.rotNet = nn.Linear(dim * 16, 2)
        # self.rotNet = nn.Linear(dim * 16, 1)
        self.locNet = nn.Linear(dim * 16, 3)

        self.dropout = nn.Dropout()

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        if self.feat:
            return x

        x = self.dropout(x)
        
        scale = self.scaleNet(x)
        rot = self.rotNet(x)
        loc = self.locNet(x)
        
        return scale, rot, loc

class StepNet(nn.Module):
    def __init__(self):
        super(StepNet, self).__init__()

        dim = 756

        self.L1 = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.predictor = Predictor(144)
        self.dropout = nn.Dropout()

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        bsz = x.size(0)
        x1 = x.sum(dim=-1)
        x2 = x.sum(dim=-2)

        x12 = torch.cat((x1, x2), axis=-1)

        x12 = self.L1(x12)

        x12 = x12.reshape(bsz, -1)

        return self.predictor(x12)
    
    def step(self, x1, x2, y):
        scale, rot, loc = self.forward(x2)

        scale, loc = y[:, :3], y[:, 4:]

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.cosine_similarity(rot, radian2vec(y[:, 3: 4])).mean()
        L3 = self.criterion(loc, y[:, 4:])

        rot = vec2radian(rot)
        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

class ConvNetV2(nn.Module):
    def __init__(self):
        super(ConvNetV2, self).__init__()

        dim = 64

        self.conv1 = nn.Conv3d(15, dim, (3, 3, 3), 2)
        self.conv2 = nn.Conv3d(dim, dim * 2, (3, 3, 3), 2)
        
        self.scaleNet = nn.Linear(dim * 16, 3)
        self.rotNet = nn.Linear(dim * 16, 2)
        self.locNet = nn.Linear(dim * 16, 3)

        self.dropout = nn.Dropout()

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = x[:, :15]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        
        scale = self.scaleNet(x)
        rot = self.rotNet(x)
        loc = self.locNet(x)
        
        return scale, rot, loc
    
    def step(self, x, y):
        scale, rot, loc = self.forward(x)

        scale = y[:, :3]
        loc = y[:, 4:]

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.cosine_similarity(rot, radian2vec(y[:, 3: 4])).mean()
        L3 = self.criterion(loc, y[:, 4:])

        rot = vec2radian(rot)
        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

class SplitNet(nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()

        self.model3d = ConvNet_2d()
        self.model2d = StepNet()

        self.criterion = torch.nn.MSELoss()

    def forward(self, x1, x2):
        scale, _ ,loc = self.model3d(x2)
        _, rot, _ = self.model2d(x2)

        return scale, rot, loc

    def step(self, x1, x2, y, z1=None, z2=None):
        scale, rot, loc = self.forward(x1, x2)
        # print (scale, y[:, :3])
        # print (loc, y[:, 4:])
        if z1 is not None:
            scale[..., 2] = z1
            loc[..., 2] = z2
        # print (scale, y[:, :3])
        # print (loc, y[:, 4:])
        # input()
        # scale = y[:, :3]
        # loc = y[:, 4:]

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.cosine_similarity(rot, radian2vec(y[:, 3: 4])).mean()
        # L2 = -torch.abs(torch.cosine_similarity(rot, radian2vec(y[:, 3: 4]))).mean()
        L3 = self.criterion(loc, y[:, 4:])

        rot = vec2radian(rot)
        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv3d(15, 64, (3, 3, 3), 2)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), 2)
        self.L1 = nn.Linear(1024, 7)

        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.L1(x)
        
        return x
    
    def step(self, x, y):
        x = self.forward(x)

        L1 = self.criterion(x[:, :3], y[:, :3])
        L2 = self.criterion(x[:, 3], y[:, 3])
        L3 = self.criterion(x[:, 4:], y[:, 4:])

        return (L1, L2, L3), x
    


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