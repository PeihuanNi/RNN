import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.upsample = nn.Linear(num_classes, 100)
        self.downsample = nn.Linear(100, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.downsample(x)
        # output = self.softmax(x)
        return x