import torch
import torch.nn as nn
from utils.utils import weights_init
from .discriminator import Discriminator



class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D = 3, ndf = 16, n_layers = 3, downsampling_factor = 4, disc_out = 512):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = Discriminator(
                ndf, n_layers, downsampling_factor, disc_out
            )

        self.downsample = nn.AvgPool1d(downsampling_factor, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


if __name__ == '__main__':
    model = MultiScaleDiscriminator()
    '''
    MultiScaleDiscriminator(
          (model): ModuleDict(
            (disc_0): Discriminator(
              (discriminator): ModuleDict(
                (layer_0): Sequential(
                  (0): ReflectionPad1d((7, 7))
                  (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_1): Sequential(
                  (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_2): Sequential(
                  (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_3): Sequential(
                  (0): Conv1d(256, 512, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_4): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_5): Conv1d(512, 1, kernel_size=(3,), stride=(1,), padding=(1,))
              )
            )
            (disc_1): Discriminator(
              (discriminator): ModuleDict(
                (layer_0): Sequential(
                  (0): ReflectionPad1d((7, 7))
                  (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_1): Sequential(
                  (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_2): Sequential(
                  (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_3): Sequential(
                  (0): Conv1d(256, 512, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_4): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_5): Conv1d(512, 1, kernel_size=(3,), stride=(1,), padding=(1,))
              )
            )
            (disc_2): Discriminator(
              (discriminator): ModuleDict(
                (layer_0): Sequential(
                  (0): ReflectionPad1d((7, 7))
                  (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_1): Sequential(
                  (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_2): Sequential(
                  (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_3): Sequential(
                  (0): Conv1d(256, 512, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_4): Sequential(
                  (0): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (layer_5): Conv1d(512, 1, kernel_size=(3,), stride=(1,), padding=(1,))
              )
            )
          )
          (downsample): AvgPool1d(kernel_size=(4,), stride=(2,), padding=(1,))
        )
        
    Length of features :  5
    Length of score :  3
    torch.Size([3, 16, 22050])
    torch.Size([3, 64, 5513])
    torch.Size([3, 256, 1379])
    torch.Size([3, 512, 345])
    torch.Size([3, 512, 345])
    torch.Size([3, 1, 345])
    Length of features :  5
    Length of score :  3
    torch.Size([3, 16, 11025])
    torch.Size([3, 64, 2757])
    torch.Size([3, 256, 690])
    torch.Size([3, 512, 173])
    torch.Size([3, 512, 173])
    torch.Size([3, 1, 173])
    Length of features :  5
    Length of score :  3
    torch.Size([3, 16, 5512])
    torch.Size([3, 64, 1378])
    torch.Size([3, 256, 345])
    torch.Size([3, 512, 87])
    torch.Size([3, 512, 87])
    torch.Size([3, 1, 87])
    4354998
        
    '''

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