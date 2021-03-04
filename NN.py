import torch
from torch import nn

class Aggregator(nn.Module):
    """Model for aggregating all input features

    Args:
        cat (nn.Module): categorical embedding model
        cont (nn.Module): continuous encoding model
        img (nn.Module): image encoding model
    """

    def __init__(self, cat, cont, img, h_dim=100):
        self.name = 'Main Neural Network'
        super(Aggregator, self).__init__()
        self.cat = cat
        self.cont = cont
        self.img = img
        self.agg_dim = sum([model.out_dim for model in [cat, cont, img]])
        self.h_dim = h_dim
        self.fc = self.fc_block(self.agg_dim, self.h_dim)
        self.out = nn.Linear(self.h_dim, 1)

    def forward(self, x):
        # x_img = torch.mean([self.img(x) for im in x['img']])
        x_img = self.img(x['img'])
        x = [self.cat(x['cat']), self.cont(x['cont']), x_img]
        x = torch.cat(x, 1)
        x = self.fc(x)
        x = self.out(x)
        return x

    def fc_block(self, in_dim, out_dim, p_drop=0.5):
        fc = nn.Linear(in_dim, out_dim)
        bn = nn.BatchNorm1d(out_dim)
        relu = nn.ReLU(True)
        dropout = nn.Dropout(p_drop)
        return nn.Sequential(*[fc, bn, relu, dropout])
