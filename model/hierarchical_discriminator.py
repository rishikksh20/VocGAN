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

    def forward(self, x, mel):
        results = []
        multi_scale_out = self.multiscale_discriminator(x, mel)
        sample_rate = hp.audio.sample_rate
        i = 1
        for key, disc in self.model.items():
            new_sample_rate = (sample_rate // (2**i))
            downsample_ = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(x)
            results.append(disc(downsample_, mel)) # [[uncond, cond], [uncond, cond], [uncond, cond], [uncond, cond]]
            i = i + 1
        return results, multi_scale_out