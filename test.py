import argparse

import numpy as np
import torch
from data_process.build_data import build_ninapro_db1,build_ninapro_db2,build_POPANE,build_RAVDESS,build_Ninapro_db5
from models.network import NinaPro_db1_Net,NinaPro_db2_Net,POPANE_Net,RAVDESS_Net,NinaPro_db5_Net
from modules.neuron import LIF,IF
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='ravdess', type=str, help='dataset name',
                        choices=['ninapro_db5', 'popane', 'ravdess'])
    parser.add_argument('--arch', default='RAVDESS_Net', type=str, help='arch name',
                        choices=['NinaPro_db5_Net', 'POPANE_Net', 'RAVDESS_Net'])
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--neuron', default='IF', type=str, help='snn type')
    parser.add_argument('--logvar_t', default=-1.0, type=float, help='logvar_t')
    parser.add_argument('--loss_type', default='HHOIB', type=str, help='loss',
                        choices=['IB', 'nlIB', '2OIB', 'ce', 'HHOIB'])

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.neuron == 'LIF':
        sn = LIF
    else:
        sn =IF
    if args.dataset == 'ninapro_db5':
        _,test_loader = build_Ninapro_db5(args.batch_size)
        num_classes = 52
    elif args.dataset == 'popane':
        _,test_loader = build_POPANE(args.batch_size)
        num_classes = 6
    elif args.dataset == 'ravdess':
        _,test_loader = build_RAVDESS(args.batch_size)
        num_classes = 4
    else:
        raise NotImplementedError('other datasets has not been completed')

    if args.arch == 'NinaPro_db5_Net':
        snn = NinaPro_db5_Net(num_classes=num_classes, sn=sn, logvar_t=args.logvar_t,
                              train_logvar_t=False, record_T=False, record_MI=False)
    elif args.arch == 'POPANE_Net':
        snn = POPANE_Net(num_classes=num_classes, sn=sn, logvar_t=args.logvar_t,
                         train_logvar_t=False, record_T=False, record_MI=False)
    elif args.arch == 'RAVDESS_Net':
        snn = RAVDESS_Net(num_classes=num_classes, sn=sn, logvar_t=args.logvar_t,
                          train_logvar_t=False, record_T=False, record_MI=False)
    else:
        raise NotImplementedError('other networks has not been completed')


    snn.to(device)
    snn.load_state_dict(torch.load(''))
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
            _, predicted = outputs.cpu().max(1)
            total += (targets.size(0))
            correct += (predicted.eq(targets.cpu()).sum().item())

    true_labels = targets.cpu().numpy()
    # 计算每个类别的准确率
    class_accuracies = {}
    for label in set(true_labels):
        indices = [i for i, t in enumerate(true_labels) if t == label]
        class_true_labels = [true_labels[i] for i in indices]
        class_predicted_labels = [predicted[i] for i in indices]
        class_accuracy = accuracy_score(class_true_labels, class_predicted_labels)
        class_accuracies[label] = round(round(class_accuracy, 4)*100,2)

    # 打印每个类别的准确率
    print("Per-class Accuracy:")
    for label, accuracy in class_accuracies.items():
        print(f"Class {label+1}: {accuracy}")

    acc = 100 * correct / total
    print(f'Avg Accuracy: {round(acc,2)}')

