import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str
from denoiser import Denoiser

MAX_WAV_VALUE = 32768.0


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)

    with torch.no_grad():
        mel = torch.from_numpy(np.load(args.input))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        audio = model.inference(mel)

        audio = audio.squeeze(0)  # collapse all dimension except time axis
        if args.d:
            denoiser = Denoiser(model).cuda()
            audio = denoiser(audio, 0.01)
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        audio = audio.cpu().detach().numpy()

        out_path = args.input.replace('.npy', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
        write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    parser.add_argument('-d', action='store_true', help="denoising ")
    args = parser.parse_args()

    main(args)
