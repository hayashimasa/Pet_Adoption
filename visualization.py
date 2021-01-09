import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from NN import Aggregator
from embedding import Categorical_Embedding
from encoder import Image_Encoder, Continuous_Encoder
from petdata import PetDataset



# x = TSNE.fit_transform(X)


def plot(epoch, metric, title, ylabel, xlabel='Epoch', label=None, color='indigo'):
    # fig = plt.figure()
    label = label if label else title
    plt.plot(epoch, metric, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def load_model(model_name):
    model_path = os.path.join(os.getcwd(), f'models/{model_name}')
    model_dict = torch.load(model_path)
    cat = Categorical_Embedding(
        model_dict['in_dim']['cat'],
        model_dict['out_dim']['cat']
    )
    cont = Continuous_Encoder(
        model_dict['in_dim']['cont'],
        model_dict['out_dim']['cont']
    )
    img = Image_Encoder(model_dict['out_dim']['img'])
    model = Aggregator(cat, cont, img)
    model.load_state_dict(model_dict['model_state_dict'])
    return model_dict, model


model_dict, model = load_model('NN_e25_lr5e-04_small.pt')


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])
# data_train = PetDataset(data_type='train', shuffle=False, small=False, image_transform=image_transform, oe=model_dict['cat_enc'])
# data_test = PetDataset(data_type='test', indicies=data_train.indicies, oe=model_dict['cat_enc'])

# print(model)

def plot_categorical_embedding(model, dataset, dim=2):
    catnet = model.cat
    catnet.eval()
    data = DataLoader(dataset)
    emb = [catnet(X['cat']) for (X, y) in data]
    emb = torch.cat(emb, 0).detach().numpy()
    emb = TSNE(dim).fit_transform(emb)
    return emb
# print(emb)
# print(emb.shape)

# plt.plot(emb)

# data_test
