import argparse
import torch
from data_process.build_data import build_ninapro_db1,build_ninapro_db2,build_POPANE,build_RAVDESS,buil_IEMOCAP,build_Ninapro_db5
from models.network import NinaPro_db1_Net,NinaPro_db2_Net,POPANE_Net,RAVDESS_Net,NinaPro_db5_Net
from utils.loss_calculate import get_loss
from modules.neuron import LIF,IF
import os
import shutil
import numpy as np
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='popane', type=str, help='dataset name',
                        choices=['ninapro_db5', 'ravdess','popane'])
    parser.add_argument('--arch', default='POPANE_Net', type=str, help='arch name',
                        choices=['NinaPro_db5_Net','RAVDESS_Net','POPANE_Net'])
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--stop_lim', default=50, type=int, help='no improvement needs stop')
    parser.add_argument('--neuron', default='IF', type=str, help='snn type',
                        choices=['IF', 'LIF'])
    parser.add_argument('--beta', default=0.5, type=float, help='beta')
    parser.add_argument('--logvar_t', default=-1.0, type=float, help='logvar_t')
    parser.add_argument('--train_logvar_t', default=False, type=bool, help='train logvar_t or not')
    parser.add_argument('--record_T', default=True, type=bool, help='record_T')
    parser.add_argument('--record_MI', default=False, type=bool, help='record_MI')
    parser.add_argument('--rerecord', default=True, type=bool, help='rerecord')#删掉results重新记录
    parser.add_argument('--save', default=True, type=bool, help='save model')
    parser.add_argument('--loss_type', default='HHOIB', type=str, help='loss',
                        choices=['IB', 'nlIB', '2OIB', 'ce', 'HHOIB'])
    args = parser.parse_args()
    if args.rerecord:
        try:
            shutil.rmtree('results')
        except:
            print('"results" dir not exists')
    mkdir('results/figure/cluster/T')
    mkdir('results/figure/cluster/Y')
    mkdir('results/logs/IXT_train')
    mkdir('results/logs/IYT_train')
    mkdir('results/logs/HY_given_T_train')
    mkdir('results/logs/IXT_test')
    mkdir('results/logs/IYT_test')
    mkdir('results/logs/HY_given_T_test')
    mkdir('results/raw')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.neuron == 'LIF':
        sn = LIF
    else:
        sn =IF
    if args.dataset == 'ninapro_db5':
        train_loader, test_loader= build_Ninapro_db5(args.batch_size)
        num_classes = 52
    elif args.dataset == 'ravdess':
        train_loader, test_loader = build_RAVDESS(args.batch_size)
        num_classes = 4
    elif args.dataset == 'popane':
        train_loader, test_loader = build_POPANE(args.batch_size)
        num_classes = 6
    else:
        raise NotImplementedError('other datasets has not been completed')

    if args.arch == 'NinaPro_db5_Net':
        snn = NinaPro_db5_Net(num_classes=num_classes,sn=sn,logvar_t=args.logvar_t,
                              train_logvar_t=args.train_logvar_t,record_T = args.record_T,record_MI = args.record_MI)
    elif args.arch == 'RAVDESS_Net':
        snn = RAVDESS_Net(num_classes=num_classes, sn=sn, logvar_t=args.logvar_t,
                              train_logvar_t=args.train_logvar_t, record_T=args.record_T, record_MI=args.record_MI)
    elif args.arch == 'POPANE_Net':
        snn = POPANE_Net(num_classes=num_classes, sn=sn, logvar_t=args.logvar_t,
                              train_logvar_t=args.train_logvar_t, record_T=args.record_T, record_MI=args.record_MI)
    else:
        raise NotImplementedError('other networks has not been completed')

    model_save_name = f'results/raw/{args.dataset}_{args.loss_type}'
    if args.loss_type != 'ce':
        model_save_name += f'_{args.beta}'

    snn.to(device)

    optimizer = torch.optim.Adam(params=snn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    accumulated = 0
    IXT_train = []
    IYT_train = []
    HY_given_T_train = []
    IXT_test = []
    IYT_test = []
    HY_given_T_test = []
    for epoch in range(args.epochs):
        snn.train()
        snn.is_eval = False
        if args.record_MI:
            IXT_train.append(0)
            IYT_train.append(0)
            HY_given_T_train.append(0)
            IXT_test.append(0)
            IYT_test.append(0)
            HY_given_T_test.append(0)
        correct = 0
        total = 0
        acc = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device).to(torch.long)
            images = images.to(device)
            dim = images.dim()
            images = images.repeat(args.step, *(1 for _ in range(dim)))
            outputs = snn(images)
            loss, IXT, IYT, HY_given_T = get_loss(snn, outputs, labels, beta=args.beta,loss_type=args.loss_type)
            if args.record_MI:
                IXT_train[epoch] += IXT.clone().detach().cpu()
                IYT_train[epoch] += IYT.clone().detach().cpu()
                HY_given_T_train[epoch] += HY_given_T.clone().detach().cpu()
            if (i + 1) % 50 == 0:
                print("Loss: ", loss)
            loss.backward()
            optimizer.step()
            total += (labels.size(0))
            _, predicted = outputs.clone().detach().cpu().max(1)
            correct += (predicted.eq(labels.cpu()).sum().item())
        scheduler.step()

        acc = 100 * correct / total
        print(f'Train Accuracy: {acc}')
        if args.record_MI:
            IXT_train[epoch] /= total
            IYT_train[epoch] /= total
            HY_given_T_train[epoch] /= total

        # start testing
        correct = 0
        total = 0
        acc = 0
        snn.eval()
        snn.is_eval = True
        with (torch.no_grad()):
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                dim = inputs.dim()
                inputs = inputs.repeat(args.step, *(1 for _ in range(dim)))
                targets = targets.to(device).to(torch.long)
                outputs = snn(inputs)

                loss, IXT, IYT, HY_given_T = get_loss(snn, outputs, targets, beta=args.beta,loss_type=args.loss_type)
                if args.record_MI:
                    IXT_test[epoch] += IXT.clone().detach().cpu()
                    IYT_test[epoch] += IYT.clone().detach().cpu()
                    HY_given_T_test[epoch] += HY_given_T.clone().detach().cpu()
                if args.record_T:
                    visualize_t = snn.T

                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        if args.record_MI:
            IXT_test[epoch] /= total
            IYT_test[epoch] /= total
            HY_given_T_test[epoch] /= total
        acc = 100 * correct / total
        print(f'Test Accuracy: {acc}')
        accumulated += 1
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            accumulated = 0
            if args.save == True:
                torch.save(snn.state_dict(), model_save_name)
        print(f'best_acc is: {best_acc}')
        print(f'best_iter: {best_epoch}')
        print(f'Iters: {epoch + 1}\n')
        if args.record_T:
            if epoch % 5 == 0:
                #每5个epoch更新一下聚类图
                np.save(f'results/figure/cluster/T/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}',visualize_t.cpu())
                np.save(f'results/figure/cluster/Y/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', targets.cpu())
        if args.record_MI:
            if epoch % 10 == 0:
                np.save(f'results/logs/IXT_train/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', IXT_train)
                np.save(f'results/logs/IYT_train/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', IYT_train)
                np.save(f'results/logs/HY_given_T_train/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', HY_given_T_train)
                np.save(f'results/logs/IXT_test/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', IXT_test)
                np.save(f'results/logs/IYT_test/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', IYT_test)
                np.save(f'results/logs/HY_given_T_test/{args.dataset}_{args.loss_type}_{args.beta}_{epoch}', HY_given_T_test)
        if accumulated > args.stop_lim:
            break
