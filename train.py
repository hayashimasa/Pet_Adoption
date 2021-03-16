from collections import OrderedDict
import os
import argparse
from pprint import pprint

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import lightgbm as lgb

from petdata import PetDataset
from encoder import (
    Categorical_Embedding, Continuous_Encoder, Image_Encoder, Text_Encoder
)
from NN import Aggregator
from metric import quadratic_weighted_kappa

def parse_args():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 32)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=128, metavar='N',
        help='input batch size for testing (default: 3)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50, metavar='N',
        help='number of epochs to train (default: 50)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0005, metavar='LR',
        help='learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )
    parser.add_argument(
        '--no-tab', action='store_true', default=False,
        help='disable tabular module'
    )
    parser.add_argument(
        '--no-img', action='store_true', default=False,
        help='disable image module'
    )
    parser.add_argument(
        '--no-text', action='store_true', default=False,
        help='disable text module'
    )
    parser.add_argument(
        '--small', action='store_true', default=False,
        help='train on smaller dataset'
    )
    parser.add_argument(
        '--save', action='store_true', default=False,
        help='save the current model'
    )
    args = parser.parse_args()
    return args

def get_train_loader(batch_size, small, shuffle):
    """initialize train dataset and dataloader

    Args:
        batch_size (int): batch size of each step
        small (boolean): use small version of dataset
        shuffle (boolean): shuffle index of when generating dataset
    """
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    data = PetDataset(
        small=small,
        shuffle=shuffle,
        image_transform=image_transform
    )
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=6
    )
    return data_loader

def get_test_loader(batch_size, indicies, oe):
    """initialize dataset and dataloader

    Args:
        batch_size (int): batch size of each step
        indicies (list(tuple(int))): indicies of training and validation set
        oe (sklearn.preprocessing.OrdinalEncoder): ordinal encoder
            obtained from training set
    """
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    data = PetDataset(
        data_type='test',
        indicies=indicies,
        oe=oe,
        image_transform=image_transform
    )
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=6
    )
    return data_loader

def train(model, data_loader, optimizer, criterion, epoch, device):
    """Train NN model for one epoch

    Args:
        model (nn.Module): model to train
        data_loader (DataLoader):
        optimizer (nn.optim.Optimizer): Optimization method
        criterion (nn.Loss): Loss function
        epoch (int): current epoch number
        device (string): device to train model ('cuda' or 'cpu')
    """
    log_interval = 4
    running_loss = 0.
    count = 0
    model.train()
    for step, (sample, target) in enumerate(data_loader):
        X, y = sample, target.view(-1, 1)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * target.shape[0]
        count += target.shape[0]

        if (step + 1) % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (step+1) * len(y), len(data_loader.dataset),
                100. * (step+1) / len(data_loader), running_loss / count)
            )
            running_loss = 0.
            count = 0
    return loss.item()

def validate(model, data_loader, criterion, device):
    """Validate NN model with validation set and print metric

    Args:
        model (nn.Module): model to validate
        data_loader (DataLoader): data iterator
        criterion (nn.Loss): Loss function
        device (string): device to train model ('cuda' or 'cpu')
    """
    model.eval()
    loss = 0.
    kappa = 0.
    n = len(data_loader.dataset)
    with torch.no_grad():
        for step, (sample, target) in enumerate(data_loader):
            X, y = sample, target.view(-1, 1)
            y_pred = model(X)
            loss += criterion(y_pred, y).item() * y.shape[0] / n
            kappa += quadratic_weighted_kappa(y_pred, y) * y.shape[0] / n

    print('\nValidation set: \n\tAverage loss: {:.4f},'.format(loss))
    print('\tquadratic weighted kappa: {:4f}\n'.format(kappa))
    return loss, kappa

def initialize_model(args, dataset):
    emb_dim = emb_size = OrderedDict([
        ('Type', 1),
        ('Name', 100),
        ('Breed1', 50),
        ('Breed2', 50),
        ('Gender', 2),
        ('Color1', 3),
        ('Color2', 3),
        ('Color3', 3),
        ('Vaccinated', 2),
        ('Dewormed', 2),
        ('Sterilized', 2),
        ('Health', 2),
        ('State', 5),
        ('RescuerID', 100),
        ('PetID', 100)
    ])
    cat_dim = dataset.cat_dim
    feature_emb = [(cat_dim[col], e_dim) for col, e_dim in emb_dim.items()]
    # Initialize model checkpoint
    model_dict = {
        'total_epoch': args.epochs,
        'in_dim': {
            'cont': dataset.cont_dim,
            'cat': feature_emb,
            'text': dataset.text_dim
        },
        'out_dim': {'cat': 100, 'cont': 5, 'img': 500, 'text': 100},
        'cat_enc': dataset.oe,
        'models': {
            'cat': True if not args.no_tab else False,
            'cont': True if not args.no_tab else False,
            'img': True if not args.no_img else False,
            'text': True if not args.no_text else False,
        },
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': list(),
        'test_loss': list(),
        'metrics': {
            'kappa': list(),
            'best': {
                'loss': None,
                'kappa': None,
                'epoch': 0
            }
        }
    }
    return model_dict

def get_model(args, dataset):
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model(args, dataset)
    cat = Categorical_Embedding(
        model_dict['in_dim']['cat'],
        model_dict['out_dim']['cat']
    ) if model_dict['models']['cat'] else None
    cont = Continuous_Encoder(
        model_dict['in_dim']['cont'],
        model_dict['out_dim']['cont']
    ) if model_dict['models']['cont'] else None
    img = (
        Image_Encoder(model_dict['out_dim']['img'])
        if model_dict['models']['img'] else None
    )
    text = (
        Text_Encoder(model_dict['out_dim']['text'])
        if model_dict['models']['text'] else None
    )
    model = Aggregator(cat, cont, img, text)
    optimizer = optim.Adam(model.parameters(), args.lr)
    # optimizer = optim.SparseAdam(list(model.parameters()), args.lr)
    if args.model:
        model.load_state_dict(model_dict['model_state_dict'])
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model, optimizer, model_dict

if __name__ == '__main__':
    args = parse_args()
    device = (
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    train_loader = get_train_loader(
        batch_size=args.batch_size,
        small=args.small,
        shuffle=True
    )
    test_loader = get_test_loader(
        batch_size=args.test_batch_size,
        indicies=train_loader.dataset.indicies,
        oe=train_loader.dataset.oe
    )
    model, optimizer, model_dict = get_model(args, train_loader.dataset)
    # print(model)
    criterion = nn.MSELoss()

    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    n_epoch = start_epoch + args.epochs - 1
    model_dict['total_epoch'] = n_epoch
    model_name = '_'.join(['NN', f'e{n_epoch}', f'lr{args.lr:.0e}'])
    model_name += '_small' if args.small else ''
    print(model_name)
    pprint(model_dict)
    print(model)
    for epoch in range(start_epoch, n_epoch+1):
        train_loss = train(
            model, train_loader, optimizer, criterion, epoch, device
        )
        test_loss, test_kappa = validate(
            model, test_loader, criterion, device
        )
        model_dict['train_loss'].append(train_loss)
        model_dict['test_loss'].append(test_loss)
        model_dict['metrics']['kappa'].append(test_kappa)
        if epoch == 1 or test_kappa > model_dict['metrics']['best']['kappa']:
            model_dict['model_state_dict'] = model.state_dict()
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['metrics']['best']['loss'] = test_loss
            model_dict['metrics']['best']['kappa'] = test_kappa
            model_dict['metrics']['best']['epoch'] = epoch
        if args.save:
            torch.save(model_dict, 'models/' + model_name + '.pt')
    pprint(model_dict['metrics']['best'])
