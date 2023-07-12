from torch import optim
from optimizers import sls

def get_algorithm(name,params):
    if name=='SGD':
        print('SGD')
        return optim.SGD(params['net'],lr=0.1)
    elif name=='Adam':
        return optim.Adam(params['net'])
    elif name=='SGD+Armijo':
        return sls.SGD(params['device'],params['net'],n_batches_per_epoch=params['n_batches_per_epoch'],c=params['c'])
    elif name=='AdamW':
        return optim.AdamW(params['net'])
    elif name=='momentum':
        return optim.SGD(params['net'],lr=0.1,momentum=0.9)
    else:
        print(f'algorithm:{name} is not registered')

    
