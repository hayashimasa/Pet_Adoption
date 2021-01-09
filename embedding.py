import torch
from torch import nn

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
        self.fc = self.fc_block(D, 100, p_drop=0.3)
        self.out = self.fc_block(100, out_dim)


    def forward(self, x):
        # x = [
        #     self.emb[i](x[:,start:end])
        #     for i, (start, end) in enumerate(self.col_idx)
        # ]
        x = [e(x[:, i]) for i, e in enumerate(self.emb)]
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

if __name__ == '__main__':
    emb_dim = [
        (2, 1),
        (9060, 100),
        (176, 100),
        (135, 100),
        (3, 2),
        (7, 3),
        (7, 3),
        (6, 3),
        (3, 2),
        (3, 2),
        (3, 2),
        (3, 2),
        (14, 5),
        (5595, 100),
        (14993, 100)
    ]
    cat_emb = Categorical_Embedding(emb_dim, 100)
    print(cat_emb)
