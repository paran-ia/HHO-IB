import numpy as np

def get_loss(snn,outputs,labels,beta,loss_type):
    HY_given_T = snn.ce(outputs,labels)
    IYT = (snn.HY - HY_given_T) / np.log(2)  # in bits
    IXT1 = snn.IXT[0]
    IXT2 = snn.IXT[1]
    IXT = IXT2
    if loss_type == 'ce':
        loss = HY_given_T
    elif loss_type == 'IB':
        loss = -1.0 * (IYT - beta * IXT2)
    elif loss_type == 'nlIB':
        loss = beta * IXT2**2 - IYT

    elif loss_type == '2OIB':
        loss = HY_given_T**2 + beta*IXT2**2

    elif loss_type == 'HHOIB':
        # loss = HY_given_T + beta * (IXT1+IXT2 ** 2)

        loss = -IYT + beta * (IXT1**2 + IXT2)
        IXT = (IXT1+IXT2)/2
    else:
        raise NotImplementedError

    return loss,IXT,IYT,HY_given_T
