import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES

def img2mse(x, y, delta): return jt.mean((x - y) ** 2)

def huber(x, y, delta):
    rel = jt.abs(x - y)
    sqr = 0.5/delta*rel*rel
    return jt.ternary((rel > delta), rel-0.5*delta, sqr).mean(-1)

def opacity_loss(opacity):
    y = opacity + 1e-10
    return -y*jt.log(y)

def distortion(ws, t1, t2):

    w = ws
    # The loss incurred between all pairs of intervals.
    interval = t1 - t2
    tmid = (t1 + t2) / 2

    dut = jt.abs(tmid[..., :, None] - tmid[..., None, :])
    loss_inter = jt.sum(w * jt.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    # The loss incurred within each individual interval with itself.
    loss_intra = jt.sum(w**2 * interval, dim=-1) / 3

    return  loss_intra + loss_inter


@LOSSES.register_module()
class OpLoss(nn.Module):
    def __init__(self, regression, opacity, distortion, delta=0.):
        self.opacity_fator = opacity
        self.distortion_fator = distortion
        self.delta = delta
        if regression == 'MSE':
            print("Using MSE")
            self.reg = img2mse
        else:
            self.reg = huber
            
    def execute(self, x, target):
        return self.reg(x, target, self.delta)  