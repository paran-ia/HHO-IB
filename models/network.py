import torch
import numpy as np
import torch.nn as nn
from spikingjelly.activation_based.layer import Conv2d,Linear,AdaptiveAvgPool2d,Conv1d,AdaptiveAvgPool1d,BatchNorm1d
from modules.neuron import IF,LIF
from utils.kde import KDE_IXT_estimation
class NinaPro_db1_Net(nn.Module):#(15047, 12, 10)
    def __init__(self,sn = IF,num_classes=52,logvar_t=-2.0, train_logvar_t=True,record_T = False,record_MI = False):
        super().__init__()
        self.HY = np.log(num_classes)  # in natts
        self.ce = nn.CrossEntropyLoss()
        self.record_T = record_T#开启会在测试集上记录隐变量T
        self.is_eval = False#是否在测试阶段，在测试阶段才会记录T
        self.record_MI = record_MI#开启会在训练集和测试集上分别记录IXT,ITY,HY_given_T
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t]).cuda()
        self.expand_dim = Linear(12*10,2*15*15)
        self.conv1 = Conv2d(in_channels=2,out_channels=64,kernel_size=3,padding=1)
        self.sn1 = sn()
        self.conv2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2)
        self.sn2 = sn()
        self.conv3 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=2)
        self.sn3 = sn()
        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,stride=2)
        self.sn4 = sn()
        self.pool = AdaptiveAvgPool2d(1)
        self.linear = Linear(256, 52)
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.step_mode = 'm'
    def encoder1(self,x):
        step = x.shape[0]
        batch = x.shape[1]
        x = x.reshape(step,batch,-1)
        x = self.expand_dim(x)
        x = x.reshape(step,batch,2,15,15)
        ###############################
        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        return x

    def encoder2(self, x):
        x = self.sn3(self.conv3(x))
        x = self.sn4(self.conv4(x))
        x = self.pool(x).squeeze()
        return x
    def decoder(self,x):
        x = self.linear(x)
        return x

    def forward(self,x):
        x = self.encoder1(x)
        #x = x + self._add_noise(x)
        T1 = x.mean(0).reshape(x.shape[1], -1)
        self.IXT = []
        self.IXT.append(self._get_IXT(T1))
        x = self.encoder2(x)
        x = x + self._add_noise(x)
        T2 = x.mean(0).reshape(x.shape[1], -1)
        if self.record_T == True:
            if self.is_eval == True:
                self.T = T2
        self.IXT.append(self._get_IXT(T2))
        x = self.decoder(x)
        return x.mean(0)

    def _add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        return noise

    def _get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.logvar_t, mean_t)  # in natts
        IXT = IXT / np.log(2)  # in bits
        return IXT

class NinaPro_db2_Net(nn.Module):#(11322, 16, 10)
    def __init__(self,sn = IF,num_classes=49,logvar_t=-1.0, train_logvar_t=True,record_T = False,record_MI = False):
        super().__init__()
        self.HY = np.log(num_classes)  # in natts
        self.ce = nn.CrossEntropyLoss()
        self.record_T = record_T#开启会在测试集上记录隐变量T
        self.is_eval = False#是否在测试阶段，在测试阶段才会记录T
        self.record_MI = record_MI#开启会在训练集和测试集上分别记录IXT,ITY,HY_given_T
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t]).cuda()
        self.expand_dim = Linear(60,1*15*15)
        self.conv1 = Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        self.sn1 = sn()
        self.conv2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2)
        self.sn2 = sn()
        self.conv3 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=2)
        self.sn3 = sn()
        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,stride=2)
        self.sn4 = sn()
        self.pool = AdaptiveAvgPool2d(1)
        self.linear = Linear(256, 49)
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.step_mode = 'm'
    def encoder1(self,x):
        step = x.shape[0]
        batch = x.shape[1]
        x = x.reshape(step,batch,-1)
        x = self.expand_dim(x)
        x = x.reshape(step,batch,1,15,15)
        ###############################
        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        return x

    def encoder2(self, x):
        x = self.sn3(self.conv3(x))
        x = self.sn4(self.conv4(x))
        print(x.shape)

        x = self.pool(x)
        print(x.shape)
        x = x.squeeze()
        return x
    def decoder(self,x):
        x = self.linear(x)
        return x

    def forward(self,x):
        x = self.encoder1(x)
        #x = x + self._add_noise(x)
        T1 = x.mean(0).reshape(x.shape[1], -1)
        self.IXT = []
        self.IXT.append(self._get_IXT(T1))
        x = self.encoder2(x)
        x = x + self._add_noise(x)
        T2 = x.mean(0).reshape(x.shape[1], -1)
        if self.record_T == True:
            if self.is_eval == True:
                self.T = T2
        self.IXT.append(self._get_IXT(T2))
        x = self.decoder(x)
        return x.mean(0)

    def _add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        return noise

    def _get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.logvar_t, mean_t)  # in natts
        IXT = IXT / np.log(2)  # in bits
        return IXT

class POPANE_Net(nn.Module):#(2519, 25)
    def __init__(self,sn = IF,num_classes=6,logvar_t=-1.0, train_logvar_t=False,record_T = False,record_MI = False):
        super().__init__()
        self.HY = np.log(num_classes)  # in natts
        self.ce = nn.CrossEntropyLoss()
        self.record_T = record_T#开启会在测试集上记录隐变量T
        self.is_eval = False#是否在测试阶段，在测试阶段才会记录T
        self.record_MI = record_MI#开启会在训练集和测试集上分别记录IXT,ITY,HY_given_T
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t]).cuda()
        self.linear1 = Linear(25,600)
        self.bn1 = nn.BatchNorm1d(600)
        self.sn1 = IF()
        self.linear2 = Linear(600, 60)
        self.bn2 = nn.BatchNorm1d(60)
        self.sn2 = IF()
        self.linear3 = Linear(60, 300)
        self.bn3 = nn.BatchNorm1d(300)
        self.sn3 = IF()
        self.linear4 = Linear(300, 30)
        self.bn4 = nn.BatchNorm1d(30)
        self.sn4 = IF()
        self.linear5 = Linear(30, 6)

        for m in self.modules():
            if isinstance(m, Linear):
                m.step_mode = 'm'
    def encoder1(self,x):
        x = self.linear1(x)
        T,B,L =x.shape
        x = x.reshape(-1,L)
        x = self.bn1(x)
        x = x.reshape(T,B,L)
        x = self.sn1(x)

        x = self.linear2(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn2(x)
        x = x.reshape(T, B, L)
        x = self.sn2(x)



        return x

    def encoder2(self, x):

        x = self.linear3(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn3(x)
        x = x.reshape(T, B, L)
        x = self.sn3(x)


        x = self.linear4(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn4(x)
        x = x.reshape(T, B, L)
        x = self.sn4(x)
        return x

    def decoder(self,x):
        x = self.linear5(x)
        return x

    def forward(self,x):
        x = self.encoder1(x)
        T1 = x.mean(0)
        self.IXT = []
        self.IXT.append(self._get_IXT(T1))
        x = self.encoder2(x)
        #x = x+self._add_noise(x)
        T2 = x.mean(0)
        if self.record_T == True:
            if self.is_eval == True:
                self.T = T2
        self.IXT.append(self._get_IXT(T2))
        x = self.decoder(x)
        return x.mean(0)

    def _add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        return noise

    def _get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.logvar_t, mean_t)  # in natts
        IXT = IXT / np.log(2)  # in bits
        return IXT
class RAVDESS_Net(nn.Module):#(930, 1582)
    def __init__(self, sn=IF, num_classes=6, logvar_t=-1.0, train_logvar_t=False, record_T=False, record_MI=False):
        super().__init__()
        self.HY = np.log(num_classes)  # in natts
        self.ce = nn.CrossEntropyLoss()
        self.record_T = record_T  # 开启会在测试集上记录隐变量T
        self.is_eval = False  # 是否在测试阶段，在测试阶段才会记录T
        self.record_MI = record_MI  # 开启会在训练集和测试集上分别记录IXT,ITY,HY_given_T
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t]).cuda()
        self.linear1 = Linear(1582, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.sn1 = sn()
        self.linear2 = Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.sn2 = sn()
        self.linear3 = Linear(4096, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.sn3 = sn()
        self.linear4 = Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.sn4 = sn()
        self.linear5 = Linear(512, 4)

        for m in self.modules():
            if isinstance(m, Linear):
                m.step_mode = 'm'

    def encoder1(self, x):
        x = self.linear1(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn1(x)
        x = x.reshape(T, B, L)
        x = self.sn1(x)

        out = self.linear2(x)
        T, B, L = out.shape
        out = out.reshape(-1, L)
        out = self.bn2(out)
        out = out.reshape(T, B, L)
        out = self.sn2(out)

        out += self.linear2(x)

        return out

    def encoder2(self, x):

        x = self.linear3(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn3(x)
        x = x.reshape(T, B, L)
        x = self.sn3(x)

        x = self.linear4(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn4(x)
        x = x.reshape(T, B, L)
        x = self.sn4(x)
        return x

    def decoder(self, x):
        x = self.linear5(x)
        return x

    def forward(self, x):
        x = self.encoder1(x)
        T1 = x.mean(0)
        self.IXT = []
        self.IXT.append(self._get_IXT(T1))
        x = self.encoder2(x)
        # x = x+self._add_noise(x)
        T2 = x.mean(0)
        if self.record_T == True:
            if self.is_eval == True:
                self.T = T2
        self.IXT.append(self._get_IXT(T2))
        x = self.decoder(x)
        return x.mean(0)

    def _add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        return noise

    def _get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.logvar_t, mean_t)  # in natts
        IXT = IXT / np.log(2)  # in bits
        return IXT
class NinaPro_db5_Net(nn.Module):
    def __init__(self,sn = IF,num_classes=52,logvar_t=-1.0, train_logvar_t=False,record_T = False,record_MI = False):
        super().__init__()
        self.HY = np.log(num_classes)  # in natts
        self.ce = nn.CrossEntropyLoss()
        self.record_T = record_T#开启会在测试集上记录隐变量T
        self.is_eval = False#是否在测试阶段，在测试阶段才会记录T
        self.record_MI = record_MI#开启会在训练集和测试集上分别记录IXT,ITY,HY_given_T
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t]).cuda()
        self.linear1 = Linear(16,16*32)
        self.bn1 = nn.BatchNorm1d(16*32)
        self.sn1 = sn()
        self.linear2 = Linear(16*32, 16*8)
        self.bn2 = nn.BatchNorm1d(16*8)
        self.sn2 = sn()
        self.linear3 = Linear(16*8, 16*64)
        self.bn3 = nn.BatchNorm1d(16*64)
        self.sn3 = sn()
        self.linear4 = Linear(16*64, 16*32)
        self.bn4 = nn.BatchNorm1d(16*32)
        self.sn4 = sn()
        self.linear5 = Linear(16*32, 53)

        for m in self.modules():
            if isinstance(m, Linear):
                m.step_mode = 'm'
    def encoder1(self,x):
        x = self.linear1(x)
        T,B,L =x.shape
        x = x.reshape(-1,L)
        x = self.bn1(x)
        x = x.reshape(T,B,L)
        x = self.sn1(x)

        out = self.linear2(x)
        T, B, L = out.shape
        out = out.reshape(-1, L)
        out = self.bn2(out)
        out = out.reshape(T, B, L)
        out = self.sn2(out)

        out+=self.linear2(x)

        return out

    def encoder2(self, x):

        x = self.linear3(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn3(x)
        x = x.reshape(T, B, L)
        x = self.sn3(x)


        x = self.linear4(x)
        T, B, L = x.shape
        x = x.reshape(-1, L)
        x = self.bn4(x)
        x = x.reshape(T, B, L)
        x = self.sn4(x)
        return x

    def decoder(self,x):
        x = self.linear5(x)
        return x

    def forward(self,x):
        x = self.encoder1(x)
        T1 = x.mean(0)
        self.IXT = []
        self.IXT.append(self._get_IXT(T1))
        x = self.encoder2(x)
        #x = x+self._add_noise(x)
        T2 = x.mean(0)
        if self.record_T == True:
            if self.is_eval == True:
                self.T = T2
        self.IXT.append(self._get_IXT(T2))
        x = self.decoder(x)
        return x.mean(0)

    def _add_noise(self, mean_t):
        noise = torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        return noise

    def _get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.logvar_t, mean_t)  # in natts
        IXT = IXT / np.log(2)  # in bits
        return IXT

if __name__ == '__main__':
    net =NinaPro_db1_Net()
    data = torch.randn(4,5,12,10)
    print(net(data).shape)