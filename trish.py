from math import gamma
from numpy import log10
import numpy
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.linalg import norm
from argparse import ArgumentError
from typing import Any, List, Optional, Union
from functools import reduce

__all__ = ["TRish"]


class TRish(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        gamma_1: float = 0.9,
        gamma_2: float = 1.1,
        dampening: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        scale: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            scale=scale,
        )
        super(TRish, self).__init__(params, defaults)
        self._params = self.param_groups[0]["params"]
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.norm_d = 0
        self.scale = scale

    def trish(
        self,
        params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        scale: bool,
        gamma_1: float,
        gamma_2: float,
        lr: float,
        has_sparse_grad: bool = None,
        foreach: bool = None,
    ):
        norm_d_p_list = []
        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            norm_d_p_list.append(norm(d_p) ** 2)

        self.norm_d = torch.sqrt(sum(norm_d_p_list))

        if not scale:
            alpha = lr
        elif self.norm_d < 1 / gamma_1:
            alpha = gamma_1 * lr
        elif self.norm_d > 1 / gamma_2:
            alpha = gamma_2 * lr
        else:
            alpha = lr / self.norm_d

        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]
            param.add_(d_p, alpha=-alpha)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure : A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False
            gamma_1 = group["gamma_1"]
            gamma_2 = group["gamma_2"]

            for p in group["params"]:
                # print(p.shape)
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            self.trish(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                scale=group["scale"],
                lr=group["lr"],
                gamma_1=gamma_1,
                gamma_2=gamma_2,
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
