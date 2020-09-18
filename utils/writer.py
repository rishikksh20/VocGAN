from tensorboardX import SummaryWriter
from utils.stft import TacotronSTFT
from .plotting import plot_waveform_to_numpy, plot_spectrogram_to_numpy
import torch

class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = hp.audio.sampling_rate
        self.stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                            hop_length=hp.audio.hop_length,
                            win_length=hp.audio.win_length,
                            n_mel_channels=hp.audio.n_mel_channels,
                            sampling_rate=hp.audio.sampling_rate,
                            mel_fmin=hp.audio.mel_fmin,
                            mel_fmax=hp.audio.mel_fmax)
        self.is_first = True

    def log_training(self, g_loss, d_loss, adv_loss, step):
        self.add_scalar('train.g_loss', g_loss, step)
        self.add_scalar('train.d_loss', d_loss, step)
        self.add_scalar('train.adv_loss', adv_loss, step)

    def log_validation(self, g_loss, d_loss, adv_loss, generator, discriminator, target, prediction, step):
        self.add_scalar('validation.g_loss', g_loss, step)
        self.add_scalar('validation.d_loss', d_loss, step)
        self.add_scalar('validation.adv_loss', adv_loss, step)
        self.add_audio('raw_audio_predicted', prediction, step, self.sample_rate)
        self.add_image('waveform_predicted', plot_waveform_to_numpy(prediction), step)
        wav = torch.from_numpy(prediction).unsqueeze(0)
        mel = self.stft.mel_spectrogram(wav)  # mel [1, num_mel, T]
        self.add_image('melspectrogram_prediction', plot_spectrogram_to_numpy(mel.squeeze(0).data.cpu().numpy()),
                       step, dataformats='HWC')
        self.log_histogram(generator, step)
        self.log_histogram(discriminator, step)
        
    def log_evaluation(self, generated, step, name):
        self.add_audio(f'{name}', generated, step, self.sample_rate)

        if self.is_first:
            self.add_audio('raw_audio_target', target, step, self.sample_rate)
            self.add_image('waveform_target', plot_waveform_to_numpy(target), step)
            wav = torch.from_numpy(target).unsqueeze(0)
            mel = self.stft.mel_spectrogram(wav)  # mel [1, num_mel, T]
            self.add_image('melspectrogram_target', plot_spectrogram_to_numpy(mel.squeeze(0).data.cpu().numpy()),
                           step, dataformats='HWC')
            self.is_first = False

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
