import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from utils.utils import read_wav_np


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.path = hp.data.train if train else hp.data.validation
        self.wav_list = glob.glob(os.path.join(self.path, '**', '*.wav'), recursive=True)
        #print("Wavs path :", self.path)
        #print(self.hp.data.mel_path)
        #print("Length of wavelist :", len(self.wav_list))
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):

        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        id = os.path.basename(wavpath).split(".")[0]

        mel_path = "{}/{}.npy".format(self.hp.data.mel_path, id)

        sr, audio = read_wav_np(wavpath)
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        # mel = torch.load(melpath).squeeze(0) # # [num_mel, T]

        mel = torch.from_numpy(np.load(mel_path))

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio


def collate_fn(batch):

    sr = 22050
    # perform padding and conversion to tensor
    mels_g = [x[0][0] for x in batch]
    audio_g = [x[0][1] for x in batch]

    mels_g = torch.stack(mels_g)
    audio_g = torch.stack(audio_g)

    sub_orig_1 = torchaudio.transforms.Resample(sr, (sr // 2))(audio_g)
    sub_orig_2 = torchaudio.transforms.Resample(sr, (sr // 4))(audio_g)
    sub_orig_3 = torchaudio.transforms.Resample(sr, (sr // 8))(audio_g)
    sub_orig_4 = torchaudio.transforms.Resample(sr, (sr // 16))(audio_g)

    mels_d = [x[1][0] for x in batch]
    audio_d = [x[1][1] for x in batch]
    mels_d = torch.stack(mels_d)
    audio_d = torch.stack(audio_d)
    sub_orig_1_d = torchaudio.transforms.Resample(sr, (sr // 2))(audio_d)
    sub_orig_2_d = torchaudio.transforms.Resample(sr, (sr // 4))(audio_d)
    sub_orig_3_d = torchaudio.transforms.Resample(sr, (sr // 8))(audio_d)
    sub_orig_4_d = torchaudio.transforms.Resample(sr, (sr // 16))(audio_d)

    return [mels_g, audio_g, sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4],\
           [mels_d, audio_d, sub_orig_1_d, sub_orig_2_d, sub_orig_3_d, sub_orig_4_d]