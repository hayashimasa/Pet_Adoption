import torch
from torch import nn
from encoder import FC_Block

class Aggregator(nn.Module):
    """Model for aggregating all input features

    Args:
        cat (nn.Module): categorical embedding model
        cont (nn.Module): continuous encoding model
        img (nn.Module): image encoding model
    """

    def __init__(self, cat=None, cont=None, img=None, text=None, h_dim=100):
        self.name = 'Output Network'
        super(Aggregator, self).__init__()
        self.cat = cat
        self.cont = cont
        self.img = img
        self.text = text
        names = ['cat', 'cont', 'img', 'text']
        self.models = {
            name: getattr(self, name) for name in names
            if not getattr(self, name) is None
        }
        self.agg_dim = sum([model.out_dim for _, model in self.models.items()])
        self.h_dim = h_dim
        self.fc = FC_Block(self.agg_dim, self.h_dim)
        self.out = nn.Linear(self.h_dim, 1)

    def forward(self, x):
        # x_img = torch.mean([self.img(x) for im in x['img']])
        x = [model(x[name]) for name, model in self.models.items()]
        #     self.cat(x['cat']), self.cont(x['cont']),
        #     self.img(x['img']), self.text(x['text'])
        # ]
        x = torch.cat(x, 1)
        x = self.fc(x)
        x = self.out(x)
        return x
