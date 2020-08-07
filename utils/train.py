import os
import math
import tqdm
import torch
import itertools
import traceback
from utils.pqmf import PQMF
from model.generator import Generator
from model.hierarchical_discriminator import Heirarchical_JCU_Discriminator
from .utils import get_commit_hash
from .validation import validate
from utils.stft_loss import MultiResolutionSTFTLoss
import torchaudio

def train(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    #print("Generator : \n",model_g)

    model_d = Heirarchical_JCU_Discriminator().cuda()
    #print("Discriminator : \n", model_d)
    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()


    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        stft_loss = MultiResolutionSTFTLoss()
        criterion = torch.nn.MSELoss().cuda()

        for epoch in itertools.count(init_epoch+1):
            if epoch % hp.log.validation_interval == 0:
                with torch.no_grad():
                    validate(hp, args, model_g, model_d, valloader, stft_loss, criterion, writer, step)

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            avg_g_loss = []
            avg_d_loss = []
            avg_adv_loss = []
            for (melG, audioG), (melD, audioD) in loader:
                melG = melG.cuda()      # torch.Size([16, 80, 64])
                audioG = audioG.cuda()  # torch.Size([16, 1, 16000])
                melD = melD.cuda()      # torch.Size([16, 80, 64])
                audioD = audioD.cuda()  #torch.Size([16, 1, 16000]

                # generator
                optim_g.zero_grad()
                sub_4, sub_3, sub_2, sub_1, fake_audio = model_g(melG)  # torch.Size([16, 1, 12800])
                fake_audio = fake_audio[:, :, :hp.audio.segment_length]




                sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audioG.size(2)].squeeze(1), audioG.squeeze(1))
                loss_g = sc_loss + mag_loss

                adv_loss = 0.0
                sample_rate = hp.audio.sampling_rate
                if step > hp.train.discriminator_train_start_steps:
                    sub_orig_1 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 2))(audioG)
                    sub_orig_2 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 4))(audioG)
                    sub_orig_3 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 8))(audioG)
                    sub_orig_4 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 16))(audioG)
                    disc_real, disc_real_multiscale = model_d([sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4], audioG, melG)
                    disc_fake, disc_fake_multiscale = model_d([sub_4, sub_3, sub_2, sub_1], fake_audio, melG)
                    # for multi-scale discriminator

                    for score_fake in disc_fake:
                        # adv_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                        adv_loss += criterion(score_fake[0], torch.ones_like(score_fake[0]))
                        adv_loss += criterion(score_fake[1], torch.ones_like(score_fake[1]))

                    for score_fake in disc_fake_multiscale:
                        # adv_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                        adv_loss += criterion(score_fake[0], torch.ones_like(score_fake[0]))
                        adv_loss += criterion(score_fake[1], torch.ones_like(score_fake[1]))

                    adv_loss = 0.5 * adv_loss

                    loss_feat = 0
                    feat_weights = 4.0 / (2 + 1) # Number of downsample layer in discriminator = 2
                    D_weights = 1.0 / 7.0 # number of discriminator = 7
                    wt = D_weights * feat_weights
                    if hp.model.feat_loss :
                        for feats_fake, feats_real in zip(disc_fake, disc_real):
                            loss_feat += wt * torch.mean(torch.abs(feats_fake[0] - feats_real[0]))
                            loss_feat += wt * torch.mean(torch.abs(feats_fake[1] - feats_real[1]))
                        for feats_fake, feats_real in zip(disc_fake_multiscale, disc_real_multiscale):
                            loss_feat += wt * torch.mean(torch.abs(feats_fake[0] - feats_real[0]))
                            loss_feat += wt * torch.mean(torch.abs(feats_fake[1] - feats_real[1]))

                    loss_g += hp.model.lambda_adv * adv_loss + hp.model.feat_match * loss_feat
            

                loss_g.backward()
                optim_g.step()

                # discriminator
                loss_d_avg = 0.0
                if step > hp.train.discriminator_train_start_steps:
                    sub_4, sub_3, sub_2, sub_1, fake_audio = model_g(melD)
                    fake_audio = fake_audio[:, :, :hp.audio.segment_length]
                    sub_4, sub_3, sub_2, sub_1, fake_audio = sub_4.detach(), sub_3.detach(), sub_2.detach(), sub_1.detach(), fake_audio.detach()
                    loss_d_sum = 0.0
                    for _ in range(hp.train.rep_discriminator):
                        optim_d.zero_grad()
                        disc_fake, disc_fake_multiscale = model_d([sub_4, sub_3, sub_2, sub_1], fake_audio, melD)
                        sub_orig_1 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 2))(audioD)
                        sub_orig_2 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 4))(audioD)
                        sub_orig_3 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 8))(audioD)
                        sub_orig_4 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 16))(audioD)
                        disc_real, disc_real_multiscale = model_d([sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4], audioD, melD)
                        loss_d_real = 0.0
                        loss_d_fake = 0.0
                        for score_fake, score_real in zip(disc_fake, disc_real):
                            loss_d_real += criterion(score_real[0], torch.ones_like(score_real[0])) # Unconditional
                            loss_d_real += criterion(score_real[1], torch.ones_like(score_real[1])) # Conditional
                            loss_d_fake += criterion(score_fake[0], torch.zeros_like(score_fake[0]))
                            loss_d_fake += criterion(score_fake[1], torch.zeros_like(score_fake[1]))

                        for score_fake, score_real in zip(disc_fake_multiscale, disc_real_multiscale):
                            loss_d_real += criterion(score_real[0], torch.ones_like(score_real[0]))
                            loss_d_real += criterion(score_real[1], torch.ones_like(score_real[1]))
                            loss_d_fake += criterion(score_fake[0], torch.zeros_like(score_fake[0]))
                            loss_d_fake += criterion(score_fake[1], torch.zeros_like(score_fake[1]))

                        loss_d_real = 0.5 * loss_d_real #/ len(disc_real) # len(disc_real) = 3
                        loss_d_fake = 0.5 * loss_d_fake #/ len(disc_fake) # len(disc_fake) = 3
                        loss_d = loss_d_real + loss_d_fake
                        loss_d.backward()
                        optim_d.step()
                        loss_d_sum += loss_d
                    loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                    loss_d_avg = loss_d_avg.item()

                step += 1
                # logging
                loss_g = loss_g.item()
                avg_g_loss.append(loss_g)
                avg_d_loss.append(loss_d_avg)
                avg_adv_loss.append(adv_loss)

                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    logger.error("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d_avg, adv_loss, step)
                    loader.set_description("Avg : g %.04f d %.04f ad %.04f| step %d" % (sum(avg_g_loss) / len(avg_g_loss),
                                                                                sum(avg_d_loss) / len(avg_d_loss),
                                                                                sum(avg_adv_loss) / len(avg_adv_loss),
                                                                                step))
            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                    % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
