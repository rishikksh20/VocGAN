import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ndf = 16, n_layers = 3, downsampling_factor = 4, disc_out = 512):
        super(Discriminator, self).__init__()
        discriminator = nn.ModuleDict()
        discriminator["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.utils.weight_norm(nn.Conv1d(1, ndf, kernel_size=15, stride=1)),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, disc_out)

            discriminator["layer_%d" % n] = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                )),
                nn.LeakyReLU(0.2, True),
            )
        nf = min(nf * 2, disc_out)
        discriminator["layer_%d" % (n_layers + 1)] = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(nf, disc_out, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )

        discriminator["layer_%d" % (n_layers + 2)] = nn.utils.weight_norm(nn.Conv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ))
        self.discriminator = discriminator

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for key, module in self.discriminator.items():
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]

# JCU Discriminator
class JCU_Discriminator(nn.Module):
    def __init__(self):
        super(JCU_Discriminator, self).__init__()
        self.mel_conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(80, 128, kernel_size=2, stride=1)),
            nn.LeakyReLU(0.2, True),
        )
        x_conv = [nn.ReflectionPad1d(7),
            nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=7, stride=1)),
            nn.LeakyReLU(0.2, True),
            ]
        x_conv += [
                        nn.utils.weight_norm(nn.Conv1d(
                                        16,
                                        64,
                                        kernel_size=41,
                                        stride=4,
                                        padding=4 * 5,
                                        groups=16 // 4,
                                    )
                                    ),
                nn.LeakyReLU(0.2),
            ]
        x_conv += [
            nn.utils.weight_norm(nn.Conv1d(
                64,
                128,
                kernel_size=21,
                stride=2,
                padding=2 * 5,
                groups=64 // 4,
            )
            ),
            nn.LeakyReLU(0.2),
        ]
        self.x_conv = nn.Sequential(*x_conv)
        self.mel_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )
        self.mel_conv3 = nn.utils.weight_norm(nn.Conv1d(
            128, 1, kernel_size=3, stride=1, padding=1
        ))

        self.x_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )
        self.x_conv3 = nn.utils.weight_norm(nn.Conv1d(
            128, 1, kernel_size=3, stride=1, padding=1
        ))

    def forward(self, x, mel):
        out = self.mel_conv(mel)
        out1 = self.x_conv(x)
        out = torch.cat([out, out1], dim=2)
        out = self.mel_conv2(out)
        cond_out = self.mel_conv3(out)
        out1 = self.x_conv2(out1)
        uncond_out = self.x_conv3(out1)
        return uncond_out, cond_out

    
if __name__ == '__main__':
    model = Discriminator()
    '''
    Length of features :  5
    Length of score :  3
    torch.Size([3, 16, 25600])
    torch.Size([3, 64, 6400])
    torch.Size([3, 256, 1600])
    torch.Size([3, 512, 400])
    torch.Size([3, 512, 400])
    torch.Size([3, 1, 400]) -> score
    '''

    x = torch.randn(3, 1, 25600)
    print(x.shape)

    features, score = model(x)
    print("Length of features : ", len(features))
    print("Length of score : ", len(score))
    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
