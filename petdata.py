from collections import OrderedDict, defaultdict
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from scipy.sparse import hstack
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from matplotlib import pyplot as plt

class PetDataset(Dataset):
    """PetFinder.my Dataset

    Args:
        csv_path (string): path of csv file containing main data
        image_path (string): path of image file containing raw image
        metadata_path (string): path of directory containing image metadata
        sentiment_path (string): path of directory containing textual
            sentiment data
        col_text (list(string)): column name of textual features
        col_cont (list(string)): column name of quantitative features
        col_p (list(string)): column name of features whose distributions
            resemble power distributions
        col_g (list(string)): column name of features whose distributions
            resemble Gaussian distributions
        target (string): target column to predict
        shuffle (bool): shuffle dataset
        data_type (string): type of data ('train' or 'test')
        n_splits (int): number of groups to split data into
        indicies (list(int)): indicies of training and testing sets
        small (bool): use a small portion of data
        oe (object): ordinal encoder from training set
    """

    def __init__(
        self, csv_path='train/train.csv', image_path='train_images/',
        metadata_path='train_metadata/', sentiment_path='train_sentiment/',
        col_txt=None, col_cont=None, col_p=None, col_g = None,
        target='AdoptionSpeed', shuffle=False, data_type='train', n_splits=5,
        indicies=None, small=False, image_transform=None, oe=None
    ):
        self.df = pd.read_csv(csv_path)
        self.image_path = image_path
        # Divide columns into groups:
        #   textual, categorical, quantitative (power/gaussian), target
        self.col_txt = ['Description'] if not col_txt else col_txt
        self.col_cont = [
            'Age', 'MaturitySize', 'FurLength', 'Quantity',
            'Fee', 'VideoAmt', 'PhotoAmt'
        ] if not col_cont else col_cont
        self.col_p = [
            'Age', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt'
        ] if not col_p else col_p
        self.col_g = ['MaturitySize'] if not col_g else col_g
        self.col_cat = [
            c for c in self.df.columns
            if c not in self.col_txt + self.col_cont + [target]
        ]
        self.target = target
        # Setup data for training
        self.data_type = data_type
        self.X = self.df.drop(columns=self.target)
        self.y = self.df[self.target]
        self.X = self.preprocess(self.X, self.col_p, self.col_g)
        self.X, self.y, self.indicies = self.split_data(
            self.X, self.y, n_splits, shuffle, indicies, data_type, small
        )
        self.X = self.make_features(self.X)
        self.oe = oe
        self.X['cat'], self.cat_dim, self.oe = self.encode(self.X['cat'])
        self.image_transform = image_transform
        self.cont_dim = len(self.col_cont)


    def preprocess(self, df, col_p, col_g):
        """Fill in null values and normalize data

        Arg:
            df (DataFrame): data to process
            col_p (list(string)): column name of features whose distributions
                resemble power distributions
            col_g (list(string)): column name of features whose distributions
                resemble Gaussian distributions
        """
        # replacing null values
        df = df.fillna(value={'Name': 'No Name', 'Description': ''})
        # normalization
        df_q = df[self.col_cont]
        med = df_q.median(axis=0)
        mu = df_q.mean(axis=0)
        sig = df_q.std(axis=0)
        for col in col_p:
            df[col] = df[col].apply(lambda x: np.log((1 + x) / (1 + med[col])))
        for col in col_g:
            df[col] = df[col].apply(lambda x: (x - mu[col]) / (sig[col]))
        return df

    def split_data(
            self, X, y, n_splits, shuffle, indicies, data_type, small=False
        ):
        """split data through Statified K-Fold
        return either training or testting data and the indicies of data

        Args:
            X (DataFrame): Input features
            y (DataFrame): target feature
            n_splits (int): number of groups to split data into
            shuffle (bool): whether or not to shuffle data before splitting
            indicies (ndarray): The training and testing set indicies
            data_type (string): 'train' or 'test'
            small (bool): generate new indicies for small dataset
        """
        # print(y.shape)
        if not indicies:
            random_state = 1 if shuffle else None
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
            indicies = list(skf.split(X, y))
        train, test = indicies[0]
        if small:
            X = X.loc[test].reset_index(drop=True)
            y = y.loc[test].reset_index(drop=True)
            X, y, small_indicies = self.split_data(
                X, y, n_splits, shuffle, None, data_type, False
            )
            indicies = [(test[idx[0]], test[idx[1]]) for idx in small_indicies]
        else:
            X = X.loc[train] if data_type == 'train' else X.loc[test]
            y = y.loc[train] if data_type == 'train' else y.loc[test]
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
        return X, y, indicies

    def make_features(self, X):
        """Reorganize data for training

        Args:
            df (DataFrame): data to reorganize
        """
        pet_imname = defaultdict(list)
        for imname in os.listdir(self.image_path):
            petid = imname.split('-')[0]
            pet_imname[petid].append(imname)
        X = {
            'cat': X[self.col_cat],
            'cont': X[self.col_cont],
            'txt': X[self.col_txt],
            'img': {idx: pet_imname[pet] for idx, pet in X['PetID'].items()}
        }
        return X

    def encode(self, X):
        """Encode categorical features with one-hot encoding

        Args:

            X (DataFrame): categorical feautres to encode
        """
        X_enc = X.copy()
        col_count = {}
        o_encoder = {} if self.oe is None else self.oe
        for col in X.columns:
            x = X_enc[col].values.reshape(-1,1)
            col_count[col] = X[col].nunique()+1
            if self.oe is None:
                oe = preprocessing.OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=col_count[col]-1
                )
                oe.fit(x)
                o_encoder[col] = oe
            else:
                oe = o_encoder[col]
            X_enc[col] = oe.transform(x)
        return X_enc, col_count, o_encoder

     # def encode(self, X):
     #    """Encode categorical features with one-hot encoding

     #    Args:
     #        X (DataFrame): categorical feautres to encode
     #    """
     #    X_enc = []
     #    col_count = {}
     #    oh_encoder = {}
     #    for col in X.columns:
     #        ohe = preprocessing.OneHotEncoder()
     #        x = X[col].values.reshape(-1,1)
     #        ohe.fit(x)
     #        X_enc.append(ohe.transform(x))
     #        # X_enc.append(x)
     #        col_count[col] = X_enc[-1].shape[1]
     #        # oh_encoder[col] = ohe

     #    return hstack(X_enc), col_count, oh_encoder

    def plot_hist(self, df=None, r=4, c=5):
        """Plot histograms of each non-textual feature

        Args:
            df (DataFrame): data to plot
        """
        df = pd.concat([self.X['cat'], self.X['cont']])#, self.y])
        fig, axs = plt.subplots(r, c, tight_layout=True, figsize=(12,8))
        col_no = ['Name', 'PetID', 'Description']
        cols = [col for col in df.columns if col not in col_no]
        for i in range(r):
            for j in range(c):
                col = cols[c*i+j]
                cl = 'darkgreen'
                if col in self.col_cont:
                    cl = 'firebrick' if col in self.col_p else 'indigo'
                axs[i, j].hist(df[col], color=cl)
                axs[i, j].set_title(col)
        # plt.show()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # print(idx)
        imnames = self.X['img'].get(idx, list())
        imgs = [Image.open(self.image_path + imname) for imname in imnames]
        imgs = [img.convert('RGB') for img in imgs]
        if not imgs:
            # initialize random image if the entry has no images
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            dim = (256, 256)
            img = torch.stack(
                [torch.normal(m, std[i], dim) for i, m in enumerate(mean)],
            )
            imgs = [transforms.ToPILImage()(img)]
        # print(imnames)
        if self.image_transform:
            imgs = [self.image_transform(img) for img in imgs]
        # print([img.shape for img in imgs])

        # if imgs and len(imgs[0].size) == 2:
        #     pass
        pet = {
            'cat': self.X['cat'].loc[idx].values.astype(np.int_),
            # 'cat': torch.from_numpy(self.X['cat'].toarray()[idx]),
            'cont': self.X['cont'].loc[idx].values.astype(np.float32),
            # 'txt': self.X['txt'].loc[idx].values,
            'img': imgs[0]# if imgs else torch.rand(3, 224, 224)
        }
        target = self.y[idx].astype(np.float32)
        return pet, target

if __name__ == '__main__':
    from torchvision import transforms

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    data_train = PetDataset(data_type='train', shuffle=False, small=False, image_transform=image_transform)
    data_test = PetDataset(data_type='test', indicies=data_train.indicies, oe=data_train.oe)
    # print(len(data_train), len(data_test))

    # print(data_train.X['cat'].columns)
    # print(data_train.X['cat'].loc[1629])
    # print(data_train[491])
    for i in range(len(data_train)):
        if data_train[i][0]['img'] is None:
            print(i, data_train[i])


    # df = pd.concat([data_train.X['cat'], data_train.X['cont'], data_train.y])
    # data_train.plot_hist()
    # data_test.plot_hist()
    # plt.show()
    # data_train.plot_hist(data_train.df)
