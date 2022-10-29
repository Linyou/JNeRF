import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS 
from jnerf.utils.config import init_cfg
from jittor.lr_scheduler import CosineAnnealingLR

@OPTIMS.register_module()
class CosAn(jt.nn.Optimizer):
    def __init__(self, nested_optimizer:jt.nn.Optimizer, T_max, eta_min=0, last_epoch=-1):

        self.sche = CosineAnnealingLR(nested_optimizer, T_max, eta_min, last_epoch)
        self._nested_optimizer = nested_optimizer
        self.steps=0

    def step(self, loss=None):
        # self.sche.last_epoch += 1
        # self.sche.update_lr()
        if self.steps % 1000 == 0:
            self.sche.last_epoch += 1
            self.sche.update_lr()
        self.steps+=1
        if loss is not None:
            self._nested_optimizer.step(loss)
