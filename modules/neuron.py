import torch.nn as nn
import torch

class LIF(nn.Module):
    def __init__(self, Vth=1, decay=0.25,reset:bool=True):
        super().__init__()
        self.Vth = Vth
        self.reset=reset
        self.decay = decay
    def forward(self, x):
        # if x.dim() != 5:
        #     x = x.unsqueeze(0)
        self.step = x.shape[0]
        mem = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            mem = mem * self.decay + x[i]
            out_s = torch.gt(mem, self.Vth)
            out_bp = torch.clamp(mem, self.Vth - 1, self.Vth + 1) * 0.5
            out_i = (out_s.float() - out_bp).detach() + out_bp

            if self.reset:
                mem = mem * (1 - out_i)
            else:
                mem = torch.where(out_i.bool(), mem - self.Vth, mem)
            out += [out_i]
        return torch.stack(out)

class IF(nn.Module):
    def __init__(self, Vth=1, reset:bool=True):
        super().__init__()
        self.Vth = Vth
        self.reset=reset
    def forward(self, x):
        # if x.dim() != 5:
        #     x = x.unsqueeze(0)
        self.step = x.shape[0]
        mem = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            mem = mem + x[i]
            out_s = torch.gt(mem, self.Vth)
            out_bp = torch.clamp(mem, self.Vth - 0.75, self.Vth + 0.75)/1.5
            out_i = (out_s.float() - out_bp).detach() + out_bp

            if self.reset:
                mem = mem * (1 - out_i)
            else:
                mem = torch.where(out_i.bool(), mem - self.Vth, mem)
            out += [out_i]
        return torch.stack(out)

if __name__ == '__main__':
    data = torch.randn((4,4,3,2,2))
    print(IF()(data).shape)