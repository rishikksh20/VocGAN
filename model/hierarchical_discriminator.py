import torch
import torch.nn as nn
from utils.utils import weights_init
from model.discriminator import JCU_Discriminator
from model.multiscale import MultiScaleDiscriminator
import torchaudio

class Heirarchical_JCU_Discriminator(nn.Module):
    def __init__(self):
        super(Heirarchical_JCU_Discriminator, self).__init__()
        self.model = nn.ModuleDict()
        self.downsample = nn.ModuleDict()
        sr = 22050
        for i in range(4):
            self.model[f"disc_{i}"] = JCU_Discriminator()
            self.downsample[f"down_{i}"] = torchaudio.transforms.Resample(sr, (sr // 2**(i+1)))

        self.multiscale_discriminator = MultiScaleDiscriminator()
        self.apply(weights_init)

    def forward(self, x, mel, sub_X = None):
        results = []
        multi_scale_out = self.multiscale_discriminator(x, mel) # D0
        i = 0
        for (key, disc), (_, down_) in zip(self.model.items(), self.downsample.items()):
            if sub_X is not None:
                x_ = sub_X[i]
            else:
                with torch.no_grad():
                    x_ = down_(x)
            results.append(disc(x_, mel)) # [[uncond, cond], [uncond, cond], [uncond, cond], [uncond, cond]]
            i = i + 1
        return results, multi_scale_out #  [D1, D2, D3, D4], D0 -> [D01, D02, D03] ,