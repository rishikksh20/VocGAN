import torch
import torch.nn as nn
from utils.utils import weights_init
from model.discriminator import JCU_Discriminator
from model.multiscale import MultiScaleDiscriminator
import torchaudio

class Heirarchical_JCU_Discriminator(nn.Module):
    def __init__(self, num_JCU_D = 4 ):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_JCU_D):
            self.model[f"disc_{i}"] = JCU_Discriminator()

        self.multiscale_discriminator = MultiScaleDiscriminator()
        self.apply(weights_init)

    def forward(self, sub_X, x, mel):
        results = []
        multi_scale_out = self.multiscale_discriminator(x, mel) # D0
        i = 1
        for key, disc in self.model.items():
            results.append(disc(sub_X[i], mel)) # [[uncond, cond], [uncond, cond], [uncond, cond], [uncond, cond]]
            i = i + 1
        return results, multi_scale_out #  [D1, D2, D3, D4], D0 -> [D01, D02, D03] ,