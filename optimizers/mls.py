import torch
import copy
import time
from torch import Tensor
from typing import List, Optional

import optimizers.utils as ut

class SGD(torch.optim.Optimizer):

    def __init__(self,
                 device,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 momentum=0,
                 weight_decay=0.,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,

                 bound_step_size=True,
                 line_search_fn="armijo"):
        defaults = dict(device=device,
                        n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        c=c,
                        beta_b=beta_b,
                        gamma=gamma,
                        beta_f=beta_f,
                        reset_option=reset_option,
                        eta_max=eta_max,
                        bound_step_size=bound_step_size,
                        line_search_fn=line_search_fn)
        super().__init__(params, defaults)       

        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        print(f'weight decay={weight_decay},momentum={momentum},c={c},reset_option={reset_option}')

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        
        
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
            
            params_current = copy.deepcopy(params)
            grad_current = ut.get_grad_list(params)
            direction_current=copy.deepcopy(grad_current)
            grad_with_weight_decay=copy.deepcopy(grad_current)
            grad_norm = ut.compute_norm(grad_current,direction_current,device=group['device'])

            step_size = ut.reset_step(step_size=batch_step_size,
                                    n_batches_per_epoch=group['n_batches_per_epoch'],
                                    gamma=group['gamma'],
                                    reset_option=group['reset_option'],
                                    init_step_size=group['init_step_size'])

            # only do the check if the gradient norm is big enough
            with torch.no_grad():
                if grad_norm >= 1e-8:
                    # check if condition is satisfied
                    found = 0
                    step_size_old = step_size

                    for e in range(100):
                        # try a prospective step
                        SGD.sgd(params,params_current,grad_current,grad_with_weight_decay,direction_current,momentum_buffer_list,weight_decay=group['weight_decay'],momentum=group['momentum'],lr=step_size)

                        # compute the loss at the next step; no need to compute gradients.
                        grad_next=ut.get_grad_list(params)
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1

                        # =================================================
                        # Line search
                        weight_norm=0.
                        weight_next_norm=0.
                        if group['weight_decay']!=0:
                            weight_norm=ut.weight_norm(params_current,group['device'])
                            weight_next_norm=ut.weight_norm(params,group['device'])

                        if group['line_search_fn'] == "armijo":
                            armijo_results = ut.check_armijo_conditions(step_size=step_size,
                                                        step_size_old=step_size_old,
                                                        loss=loss+(group['weight_decay']/2)*weight_norm,
                                                        grad_current=grad_with_weight_decay,
                                                        direction_current=direction_current,
                                                        loss_next=loss_next+(group['weight_decay']/2)*weight_next_norm,
                                                        c=group['c'],
                                                        beta_b=group['beta_b'],
                                                        device=group['device'])
                            found, step_size, step_size_old = armijo_results
                            result_step_size=step_size
                            if found == 1:
                                for p, momentum_buffer in zip(params, momentum_buffer_list):
                                    state = self.state[p]
                                    state['momentum_buffer'] = momentum_buffer
                                break


                        
                    
                    # if line search exceeds max_epochs
                    if found == 0:
                        result_step_size=1e-5
                        SGD.sgd(params,params_current,grad_current,grad_with_weight_decay,direction_current,momentum_buffer_list,weight_decay=group['weight_decay'],momentum=group['momentum'],lr=1e-6)
                        print('not found')
                        for p, momentum_buffer in zip(params, momentum_buffer_list):
                            state = self.state[p]
                            state['momentum_buffer'] = momentum_buffer
            # save the new step-size
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return result_step_size,loss,ut.compute_grad_norm(grad_current,group['device'])
    

    def sgd(params: List[Tensor],params_current: List[Tensor],
                       grad_current: List[Tensor],
                       grad_with_weight_decay:List[Tensor],
                       direction_current,
                       momentum_buffer_list: List[Optional[Tensor]],
                       weight_decay:float,
                       momentum: float,
                       lr: float,):

        for i, (param,p_current,g_current,g_with_weight_decay,d_current) in enumerate(zip(params,params_current,grad_current,grad_with_weight_decay,direction_current)):


            if weight_decay!=0 :
                grad_with_weight_decay[i]=g_current.add(p_current,alpha=weight_decay)
            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_current).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum)
                    torch.add(buf,grad_with_weight_decay[i],out=direction_current[i])


            alpha =-lr
            torch.add(p_current,g_with_weight_decay,alpha=alpha,out=param)
