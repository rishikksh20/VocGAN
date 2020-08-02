import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import weights_init
from .res_stack import ResStack
# from res_stack import ResStack

MAX_WAV_VALUE = 32768.0


class Generator(nn.Module):
    def __init__(self, mel_channel, n_residual_layers, ratios=[8, 5, 5], mult = 256, out_band = 1):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        
        generator = [nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, mult*2, kernel_size=7, stride=1)),
            ]

        # Upsample to raw audio scale
        for _, r in enumerate(ratios):
            generator += [
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.ConvTranspose1d(mult*2, mult, 
                                    kernel_size=r*2, stride=r, 
                                    padding=r // 2 + r % 2,
                                    output_padding=r % 2)
                                    ),
            ]
            for j in range(n_residual_layers):
                generator += [ResStack(mult, dilation=3 ** j)]

            mult //= 2

        generator += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mult*2, out_band, kernel_size=7, stride=1)),
            nn.Tanh(),
        ]

        self.generator = nn.Sequential(*generator)
        self.apply(weights_init)

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        return audio


'''
    to run this, fix 
    from . import ResStack
    into
    from res_stack import ResStack
'''
if __name__ == '__main__':
    '''
    torch.Size([3, 80, 10])
    torch.Size([3, 1, 2000])
    4527362
    '''
    model = Generator(80, 4)

    x = torch.randn(3, 80, 10)  # (B, channels, T).
    print(x.shape)

    y = model(x) # (B, 1, T ** prod(upsample_scales)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])  # For normal melgan torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)