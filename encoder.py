import torch
from torch import nn
from torchvision import models

class Continuous_Encoder(nn.Module):

    def __init__(self, cont_dim, out_dim):
        self.name = 'Continuous_Encoder'
        super(Continuous_Encoder, self).__init__()
        self.cont_dim = cont_dim
        self.out_dim = out_dim
        self.fc = self.fc_block(self.cont_dim, self.out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

    def fc_block(self, in_dim, out_dim, p_drop=0.5):
        fc = nn.Linear(in_dim, out_dim)
        bn = nn.BatchNorm1d(out_dim)
        relu = nn.ReLU(True)
        dropout = nn.Dropout(p_drop)
        return nn.Sequential(*[fc, bn, relu, dropout])

class Image_Encoder(nn.Module):

    def __init__(self, out_dim, im_dim=224):
        super(Image_Encoder, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        h_dim = self.encoder.fc.in_features
        self.out_dim = out_dim
        self.encoder.fc = self.fc_block(h_dim, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def fc_block(self, in_dim, out_dim, p_drop=0.5):
        fc = nn.Linear(in_dim, out_dim)
        bn = nn.BatchNorm1d(out_dim)
        relu = nn.ReLU(True)
        dropout = nn.Dropout(p_drop)
        return nn.Sequential(*[fc, bn, relu, dropout])

