import torch
import copy
import time
from torch import Tensor
from typing import List, Optional

import optimizers.utils as ut

class SGD(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 momentum=0,
                 c1=0.1,
                 c2=0.6,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="strong-wolfe"):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        momentum=momentum,
                        c1=c1,
                        c2=c2,
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

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.required_grad=True
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
            grad_current1=copy.deepcopy(grad_current)

            direction_current=copy.deepcopy(grad_current)
            grad_norm=ut.compute_grad_norm(grad_current1)
            


            step_size = ut.reset_step(step_size=batch_step_size,
                                    n_batches_per_epoch=group['n_batches_per_epoch'],
                                    gamma=group['gamma'],
                                    reset_option=group['reset_option'],
                                    init_step_size=group['init_step_size'])

            # only do the check if the gradient norm is big enough

            if grad_norm >= 1e-8:
                # check if condition is satisfied
                found = 0
                step_size_old = step_size

                for e in range(100):
                    # try a prospective step
                    with torch.no_grad():
                        SGD.sgd(params,params_current, grad_current,direction_current,momentum_buffer_list,momentum=group['momentum'],lr=step_size)
                    # compute the loss at the next step; no need to compute gradients.
                    
                    loss_next = closure_deterministic()
                    loss_next.required_grad=True
                    loss_next.backward()

                    grad_next=ut.get_grad_list(params)
                    grad_next_norm=ut.compute_grad_norm(grad_next)
                    #print('新'+str(grad_next_norm)+'旧'+str(grad_norm))

                    grad_next_t_grad_current=ut.compute_norm(grad_next,grad_current)
                    

                    self.state['n_forwards'] += 1

                    # ===============
                    # Line search
                    if group['line_search_fn'] == "strong-wolfe":
                        armijo_results = ut.check_strong_wolfe_conditions(step_size=step_size,
                                                    step_size_old=step_size_old,
                                                    loss=loss,
                                                    grad_current=grad_current,
                                                    direction_current=direction_current,
                                                    loss_next=loss_next,
                                                    c1=group['c1'],
                                                    c2=group['c2'],
                                                    beta_b=group['beta_b'],
                                                    grad_next_t_grad_current=grad_next_t_grad_current,
                                                    grad_norm=grad_norm)
                        found, step_size, step_size_old = armijo_results
                        return_step_size=step_size
                        if found == 1:
                            for p, momentum_buffer in zip(params, momentum_buffer_list):
                                state = self.state[p]
                                state['momentum_buffer'] = momentum_buffer
                            break
                    
                
                # if line search exceeds max_epochs
                if found == 0:
                    return_step_size=1e-6
                    with torch.no_grad():
                        SGD.sgd(params, params_current, grad_current,direction_current,momentum_buffer_list,momentum=group['momentum'],lr=return_step_size)
                    for p, momentum_buffer in zip(params, momentum_buffer_list):
                        state = self.state[p]
                        state['momentum_buffer'] = momentum_buffer
            # save the new step-size
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return step_size,loss
    

    def sgd(params: List[Tensor],params_current: List[Tensor],
                       grad_current: List[Tensor],
                       direction_current,
                       momentum_buffer_list: List[Optional[Tensor]],
                       momentum: float,
                       lr: float,):

        for i, (param,p_current,g_current,d_current) in enumerate(zip(params,params_current,grad_current,direction_current)):


            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_current).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum)
                    torch.add(buf,g_current,out=d_current)


            alpha =-lr
            torch.add(p_current,d_current,alpha=alpha,out=param)
