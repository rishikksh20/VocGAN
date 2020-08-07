import torch
import torch.nn as nn
from utils.utils import weights_init
from model.discriminator import JCU_Discriminator



class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D = 3, downsampling_factor = 4):
        super(MultiScaleDiscriminator, self).__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = JCU_Discriminator()

        self.downsample = nn.AvgPool1d(downsampling_factor, stride=2, padding=1, count_include_pad=False)


    def forward(self, x, mel):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, mel)) # [[uncond, cond], [uncond, cond], [uncond, cond]]
            x = self.downsample(x)
            mel = self.downsample(mel)
        return results # [D01, D02, D03]


if __name__ == '__main__':
    model = MultiScaleDiscriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)
    print(model)

    scores = model(x)
    for (features, score) in scores:
        print("Length of features : ", len(features))
        print("Length of score : ", len(score))
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)