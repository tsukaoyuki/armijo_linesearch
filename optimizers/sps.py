import torch
import copy
import time
from torch import Tensor
from typing import List, Optional

import optimizers.utils as ut

class Sps(torch.optim.Optimizer):
    def __init__(self, params,n_batches_per_epoch=500,
                c=0.5,
                gamma=2.0,
                eps=1e-8,
                momentum=0.0,
                weight_decay=0.0,
                device='cpu',):

        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
        c=c,
        gamma=gamma,
        eps=eps,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device)
        super().__init__(params, defaults)       


        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        print(f'weight decay={weight_decay},momentum={momentum},c={c}')

    def step(self, closure=None,epoch=1):
        # deterministic closure
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()
        # loop over parameter groups
        for group in self.param_groups:
            params = group["params"]
            momentum_buffer_list = []
            for p in group['params']:
                
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

            # save the current parameters:
            
            grad_current = ut.get_grad_list(params)
            direction_current=copy.deepcopy(grad_current)
            grad_with_weght_decay=copy.deepcopy(grad_current)
            grad_norm = ut.compute_norm(grad_current,direction_current,device=group['device'])
            step_size=loss / (group['c'] * (grad_norm)**2 + group['eps'])
            coeff = group['gamma']**(1./group['n_batches_per_epoch'])
            step_size = min(1,
                            step_size)
            # only do the check if the gradient norm is big enough
            with torch.no_grad():
                sgd_update(params,step_size,grad_current)

        return step_size,loss,ut.compute_grad_norm(grad_current,group['device'])


def sgd_update(params, step_size, grad_current):
    for p, g in zip(params, grad_current):
        if isinstance(g, float) and g == 0.:
            continue
        p.data.add_(other=g, alpha=- step_size)
def sgd(params: List[Tensor],
                    grad_current: List[Tensor],
                    grad_with_weght_decay:List[Tensor],
                    direction_current,
                    momentum_buffer_list: List[Optional[Tensor]],
                    weight_decay:float,
                    momentum: float,
                    lr: float,):

    for i, (param,g_current,g_with_weight_decay,d_current) in enumerate(zip(params,grad_current,grad_with_weght_decay,direction_current)):


        if weight_decay!=0 :
            g_with_weight_decay=g_current.add(p_current,alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_current).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum)
                torch.add(buf,g_with_weight_decay,out=d_current)


        alpha =-lr
        torch.add(param,g_with_weight_decay,alpha=alpha,out=param)
