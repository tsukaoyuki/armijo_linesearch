import torch
import copy
import time
from torch import Tensor
from typing import List, Optional

import optimizers.utils as ut

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,device='cuda:6'):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,device=device)
        super().__init__(params, defaults)       


        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        print(f'weight decay={weight_decay},momentum={momentum} mysgd_')
        self._params = self.param_groups[0]['params']

    def _directional_evaluate(self,closure, x, t, d):
        for i in range(len(self.params)):
            params[i]=torch.add(params[i],params[i].grad,alpha=t)
            
        loss_next = float(closure())
        grad_next=ut.get_grad_list(params)
        self._set_param(x)
        return loss_next,grad_next
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)


    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
    def _directional_evaluate(self,closure, x, t, d):
        for i,(param,grad) in enumerate(zip(x,d)):
            x[i]=torch.add(param,d[i],alpha=t)
        loss_next = float(closure())
        grad_next=ut.get_grad_list(x)
        return loss_next,grad_next
    def step(self, closure):
        # deterministic closure
        seed = time.time()
        closure = torch.enable_grad()(closure)
        orig_loss = closure()
        loss = float(orig_loss)

        
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
            x_init = self._clone_param()            
            grad_current = ut.get_grad_list(params)
            direction_current=copy.deepcopy(grad_current)
            grad_with_weght_decay=copy.deepcopy(grad_current)
            grad_norm = ut.compute_norm(grad_current,direction_current,device=group['device'])

            
            # only do the check if the gradient norm is big enough
            print(grad_norm)
            loss_next,grad_next=self._directional_evaluate(closure,params,group['lr'],grad_current)

            grad_norm_next = ut.compute_norm(grad_next,grad_next,device=group['device'])
            print(grad_norm_next)
            with torch.no_grad():
                sgd(params,grad_current,grad_with_weght_decay,direction_current,momentum_buffer_list,weight_decay=group['weight_decay'],momentum=group['momentum'],lr=group['lr'])

        return loss,ut.compute_grad_norm(grad_current,group['device'])




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
