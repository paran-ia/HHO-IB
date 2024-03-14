import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import sys
import os
np.random.seed(2024)
def build_ninapro_db1(batch_size):
    file = h5py.File('dataset/NinaPro-DB1/DB1_S1.h5','r')
    imageData = file['imageData'][:]#(15047, 12, 10)
    imageLabel = file['imageLabel'][:]#(15047,)
    file.close()
    # 随机打乱数据和标签
    N = imageData.shape[0]
    index = np.random.permutation(N)
    data = imageData[index, :, :]
    label = imageLabel[index]

    # 对数据升维,标签one-hot
    #data = np.expand_dims(data, axis=1)

    N = data.shape[0]
    num_train = round(N * 0.8)
    num_val = round((N-num_train)/2)
    num_test = N-num_train-num_val
    # print('num: ', N)

    X_train = data[0:num_train, :, :]
    Y_train = label[0:num_train]

    X_val = data[num_train:N-num_test,:,:]
    Y_val = label[num_train:N-num_test]


    X_test = data[N-num_test:N, :, :]
    Y_test = label[N-num_test:N]

    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))
    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_train, y=Y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_val, y=Y_val),
        batch_size=X_val.shape[0], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_test, y=Y_test),
        batch_size=X_test.shape[0], shuffle=False
    )
    return train_loader,val_loader,test_loader

def build_ninapro_db2(batch_size):

    file = h5py.File('dataset/NinaPro-DB2/DB2_S1_feature_200_0.h5', 'r')
    imageData = file['featureData'][:]
    imageLabel = file['featureLabel'][:]
    file.close()
    # data= loadmat('../dataset/NinaPro-DB2/data2.mat')
    # imageData = data['data']#(11322, 16, 10)
    # imageLabel = data['label']#(11322, 1)
    # print(imageData.shape)#(11914, 60)
    # print(imageLabel.shape)#(11914,)
    # 随机打乱数据和标签
    N = imageData.shape[0]
    index = np.random.permutation(N)
    data = imageData[index, :]
    label = imageLabel[index]


    # 对数据升维,标签one-hot
    #data = np.expand_dims(data, axis=1)
    # label = convert_to_one_hot(label-1, 52).T

    # 划分数据集
    N = data.shape[0]
    num_train = round(N * 0.8)
    num_val = round((N - num_train) / 2)
    num_test = N - num_train - num_val
    # print('num: ', N)
    X_train = data[0:num_train, :]
    Y_train = label[0:num_train]

    X_val = data[num_train:N - num_test, :]
    Y_val = label[num_train:N - num_test]

    X_test = data[N - num_test:N, :]
    Y_test = label[N - num_test:N]

    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_train, y=Y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_val, y=Y_val),
        batch_size=X_val.shape[0], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_test, y=Y_test),
        batch_size=X_test.shape[0], shuffle=False
    )
    return train_loader, val_loader, test_loader

def build_POPANE(batch_size):
    features = np.load('dataset/POPANE/EEG_features.npy')#(2519, 25)
    labels = np.load('dataset/POPANE/EEG_labels.npy')#(2519,)
    # 随机打乱数据和标签
    N = features.shape[0]
    index = np.random.permutation(N)
    data = features[index, :]
    label = labels[index]
    num_train = round(N * 0.8)
    # num_val = round((N - num_train) / 2)
    # num_test = N - num_train - num_val
    num_test = N - num_train
    # print('num: ', N)
    X_train = data[0:num_train, :]
    Y_train = label[0:num_train]

    # X_val = data[num_train:N - num_test, :]
    # Y_val = label[num_train:N - num_test]

    X_test = data[N - num_test:N, :]
    Y_test = label[N - num_test:N]

    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_train, y=Y_train),
        batch_size=batch_size,
        shuffle=True
    )
    # val_loader = torch.utils.data.DataLoader(
    #     Custom_Dataset(x=X_val, y=Y_val),
    #     batch_size=X_val.shape[0], shuffle=False
    # )
    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_test, y=Y_test),
        batch_size=X_test.shape[0], shuffle=False
    )
    return train_loader,test_loader

def build_RAVDESS(batch_size):
    features = np.load('dataset/RAVDESS/speech_features.npy')#(930, 1582)
    labels = np.load('dataset/RAVDESS/speech_labels.npy')#(930,)
    # 随机打乱数据和标签
    N = features.shape[0]
    index = np.random.permutation(N)
    data = features[index, :]
    label = labels[index]
    num_train = round(N * 0.9)
    num_test = N - num_train
    # print('num: ', N)
    X_train = data[0:num_train, :]
    Y_train = label[0:num_train]

    X_test = data[N - num_test:N, :]
    Y_test = label[N - num_test:N]

    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_train, y=Y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=X_test, y=Y_test),
        batch_size=X_test.shape[0], shuffle=False
    )
    return train_loader, test_loader
def build_Ninapro_db5(batch_size):
    test_features = np.load('dataset/NinaPro-DB5/EMG_test.npy')#(7582, 16)
    test_labels = np.load('dataset/NinaPro-DB5/label_test.npy')#(7582,)
    train_features = np.load('dataset/NinaPro-DB5/EMG_train.npy')#(30326, 16)
    train_labels = np.load('dataset/NinaPro-DB5/label_train.npy')#(30326,)
    N_test = test_features.shape[0]
    index_test = np.random.permutation(N_test)
    test_features = test_features[index_test,:, :].mean(1).squeeze()
    test_labels = test_labels[index_test]
    N_train = train_features.shape[0]
    index_train = np.random.permutation(N_train)
    train_features = train_features[index_train,:, :].mean(1).squeeze()
    train_labels = train_labels[index_train]
    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=train_features, y=train_labels),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=test_features, y=test_labels),
        batch_size=test_features.shape[0], shuffle=False
    )
    return train_loader, test_loader
def buil_IEMOCAP(batch_size):
    test_features = np.load('dataset/IEMOCAP/test_feature.npy')#(1960, 8)
    test_labels = np.load('dataset/IEMOCAP/test_label.npy')#(1960,)
    train_features = np.load('dataset/IEMOCAP/train_feature.npy')#(7837, 8)
    train_labels = np.load('dataset/IEMOCAP/train_label.npy')#(7837,)
    N_test = test_features.shape[0]
    index_test = np.random.permutation(N_test)
    test_features = test_features[index_test, :]
    test_labels = test_labels[index_test]
    N_train = train_features.shape[0]
    index_train = np.random.permutation(N_train)
    train_features = train_features[index_train, :]
    train_labels = train_labels[index_train]
    train_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=train_features, y=train_labels),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        Custom_Dataset(x=test_features, y=test_labels),
        batch_size=test_features.shape[0], shuffle=False
    )
    return train_loader,test_loader
class Custom_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(dtype=torch.FloatTensor)
        self.y = torch.from_numpy(y).type(dtype=torch.FloatTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, item):
        x_i = self.x[item, :]
        y_i = self.y[item]
        return x_i, y_i

    def __len__(self):
        return self.len

if __name__ == '__main__':

    # train_loader, test_loader = build_ninapro_db2(32)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print("Data shape:", data.shape)
    #     print("Target shape:", target.shape)
    build_Ninapro_db5(32)