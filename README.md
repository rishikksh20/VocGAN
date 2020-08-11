# VocGAN
Unofficial PyTorch implementation of [VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network](https://arxiv.org/abs/2007.15256).

![](./assets/vocgan.JPG)

Tested on Python 3.6
```bash
pip install -r requirements.txt
```

## Prepare Dataset

- Download dataset for training. This can be any wav files with sample rate 22050Hz. (e.g. LJSpeech was used in paper)
- preprocess: `python preprocess.py -c config/default.yaml -d [data's root path]`
- Edit configuration `yaml` file

## Train & Tensorboard

- `python trainer.py -c [config yaml file] -n [name of the run]`
  - `cp config/default.yaml config/config.yaml` and then edit `config.yaml`
  - Write down the root path of train/validation files to 2nd/3rd line.
  
- `tensorboard --logdir logs/`

## Notes
1) This is the rough implementation of the VocGAN paper as author don't provide much details regarding architecture and parameters.
2) Traning cost for `Discriminator` is too high (2.8 sec/it on P100 with batch size 16) as compared to `Generator` (7.2 it/sec on P100 with batch size 16), so it's unfeasible for me to train this model for long time.
3) May be we can optimizer Discriminator by downsampling the audio on pre-processing stage instead of Training stage (currently I used `torchaudio.transform.Resample` as layer for downsampling the audio), this step might be speed-up overall `Discriminator` training.
4) I trained this model for 300 epochs (with batch size 16) on LJSpeech, and quality of generated audio is similar to the MelGAN at same epoch on same dataset. Author recommend to train model till 3000 epochs which is not feasible at current training speed `(2.80 sec/it)`.
5) I am open for any suggestion and modification on this repo.

## Inference

- `python inference.py -p [checkpoint path] -i [input mel path]`

## Results
[WIP]

## References
- [VocGAN](https://arxiv.org/abs/2007.15256)
- [Multi-band MelGAN](https://arxiv.org/abs/2005.05106)
- [MelGAN](https://arxiv.org/abs/1910.06711)
- [Pytorch implementation of melgan](https://github.com/seungwonpark/melgan)
- [Official implementation of melgan](https://github.com/descriptinc/melgan-neurips)
- [Multi, Full-band melgan implementation](https://github.com/rishikksh20/melgan)
- [Nvidia's pre-processing](https://github.com/NVIDIA/tacotron2)
- [WaveRNN](https://github.com/fatchord/WaveRNN)

