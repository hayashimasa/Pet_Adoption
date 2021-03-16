import torch
from torch import nn
from torchvision import models

class FC_Block(nn.Module):
    """Fully connected layer with batch normalization, ReLU, and dropout
    """
    def __init__(self, in_dim, out_dim, p_drop=0.5):
        self.name = 'Fully Connected Block'
        super(FC_Block, self).__init__()
        fc = nn.Linear(in_dim, out_dim)
        bn = nn.BatchNorm1d(out_dim)
        relu = nn.ReLU(True)
        dropout = nn.Dropout(p_drop)
        self.fc_block = nn.Sequential(*[fc, bn, relu, dropout])

    def forward(self, x):
        return self.fc_block(x)


class Categorical_Embedding(nn.Module):
    """Model for embedding categorical features

    Args:
        emb_dim (list(tuple(int))): each tuples stores the input and output
            dimensions for each categorical feature
        out_dim (int): outpit dimension of the categorical embedding
    """
    def __init__(self, emb_dim, out_dim):
        self.name = 'Categorical_Embedding'
        super(Categorical_Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        M = [(0,0)] + emb_dim
        self.col_idx = [(m[0], M[i+1][0]) for i, m in enumerate(M[:-1])]
        self.emb = nn.ModuleList([nn.Embedding(m, d) for (m, d) in emb_dim])
        D = sum([d for (m, d) in emb_dim])
        self.h = FC_Block(D, 100)
        self.out = FC_Block(100, out_dim)


    def forward(self, x):
        # x = [
        #     self.emb[i](x[:,start:end])
        #     for i, (start, end) in enumerate(self.col_idx)
        # ]
        x = [e(x[:, i]) for i, e in enumerate(self.emb)]
        x = torch.cat(x, 1)
        x = self.h(x)
        x = self.out(x)
        return x

class Continuous_Encoder(nn.Module):
    """Model for encoding continuous features

    Args:
        cont_dim (int): dimension of input continuous features
        out_dim (int): outpit dimension of the encoded features
    """
    def __init__(self, cont_dim, out_dim):
        self.name = 'Continuous_Encoder'
        super(Continuous_Encoder, self).__init__()
        self.cont_dim = cont_dim
        self.h_dim = 10
        self.out_dim = out_dim
        self.h = FC_Block(self.cont_dim, self.h_dim)
        self.fc = FC_Block(self.h_dim, self.out_dim)


    def forward(self, x):
        x = self.h(x)
        x = self.fc(x)
        return x

class Image_Encoder(nn.Module):
    """Model for encoding image features

    Args:
        out_dim (int): outpit dimension of the encoded features
        im_dim (int): image size
    """
    def __init__(self, out_dim, im_dim=224):
        self.name = 'Image Encoder'
        super(Image_Encoder, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        h_dim = self.encoder.fc.in_features
        self.out_dim = out_dim
        self.encoder.fc = FC_Block(h_dim, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Text_Encoder(nn.Module):
    """Model for encoding textual features
    """
    def __init__(self, out_dim, text_dim=300):
        self.name = 'Text Encoder'
        super(Text_Encoder, self).__init__()
        self.text_dim = text_dim
        self.h_dim = 50
        self.out_dim = out_dim
        self.h = FC_Block(self.text_dim, self.h_dim)
        self.out = FC_Block(self.h_dim, self.out_dim)

    def forward(self, x):
        x = self.h(x)
        x = self.out(x)
        return x
