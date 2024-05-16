import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input images 1 * 640 * 640
        self.L1 = nn.Sequential(nn.Conv2d(1, 16, 5), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))  # 16 * 318 * 318
        self.L2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 158 * 158
        self.L3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU())  # 64 * 156 * 156
        self.L4 = nn.Sequential(nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 77 * 77
        self.L5 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 38 * 38
        self.L6 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())  # 128 * 36 * 36
        self.L7 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))  # 256 * 17 * 17
        self.L8 = nn.Sequential(nn.Conv2d(256, 256, 3), nn.BatchNorm2d(256), nn.ReLU())  # 256 * 15 * 15
        self.L9 = nn.Sequential(nn.Conv2d(256, 256, 3), nn.BatchNorm2d(256), nn.ReLU())  # 256 * 13 * 13
        self.L10 = nn.Sequential(nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))  # 512 * 6 * 6
        self.L11 = nn.Sequential(nn.Conv2d(512, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))  # 512 * 2 * 2

        self.FC1 = nn.Sequential(nn.Linear(128 * 2 * 2, 256), nn.ReLU())
        self.FC2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.FC3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        self.Last = nn.Linear(64, 4)

    def forward(self, x):
        x = self.L11(self.L10(self.L9(self.L8(self.L7(self.L6(self.L5(self.L4(self.L3(self.L2(self.L1(x)))))))))))
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.Last(x)
        return x
