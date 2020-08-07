import tqdm
import torch
import torchaudio

def validate(hp, args, generator, discriminator, valloader, stft_loss, criterion, writer, step):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for mel, audio in loader:
        mel = mel.cuda()
        audio = audio.cuda()    # B, 1, T torch.Size([1, 1, 212893])

        # generator
        sub_4, sub_3, sub_2, sub_1, fake_audio = generator(mel)  # torch.Size([16, 1, 12800])

        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss
        sample_rate = hp.audio.sampling_rate
        sub_orig_1 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 2))(audio)
        sub_orig_2 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 4))(audio)
        sub_orig_3 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 8))(audio)
        sub_orig_4 = torchaudio.transforms.Resample(sample_rate, (sample_rate // 16))(audio)
        disc_real, disc_real_multiscale = discriminator([sub_orig_1, sub_orig_2, sub_orig_3, sub_orig_4], audio, mel)
        disc_fake, disc_fake_multiscale = discriminator([sub_4, sub_3, sub_2, sub_1], fake_audio[:, :, :audio.size(2)], mel)


        adv_loss =0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0

        for score_fake, score_real in zip(disc_fake, disc_real):
            adv_loss += criterion(score_fake[0], torch.ones_like(score_fake[0]))
            adv_loss += criterion(score_fake[1], torch.ones_like(score_fake[1]))
            loss_d_real += criterion(score_real[0], torch.ones_like(score_real[0]))  # Unconditional
            loss_d_real += criterion(score_real[1], torch.ones_like(score_real[1]))  # Conditional
            loss_d_fake += criterion(score_fake[0], torch.zeros_like(score_fake[0]))
            loss_d_fake += criterion(score_fake[1], torch.zeros_like(score_fake[1]))

        for score_fake, score_real in zip(disc_fake_multiscale, disc_real_multiscale):
            adv_loss += criterion(score_fake[0], torch.ones_like(score_fake[0]))
            adv_loss += criterion(score_fake[1], torch.ones_like(score_fake[1]))
            loss_d_real += criterion(score_real[0], torch.ones_like(score_real[0]))
            loss_d_real += criterion(score_real[1], torch.ones_like(score_real[1]))
            loss_d_fake += criterion(score_fake[0], torch.zeros_like(score_fake[0]))
            loss_d_fake += criterion(score_fake[1], torch.zeros_like(score_fake[1]))

        feat_weights = 4.0 / (2 + 1)  # Number of downsample layer in discriminator = 2
        D_weights = 1.0 / 7.0  # number of discriminator = 7
        wt = D_weights * feat_weights
        loss_feat = 0
        if hp.model.feat_loss:
            for feats_fake, feats_real in zip(disc_fake, disc_real):
                loss_feat += wt * torch.mean(torch.abs(feats_fake[0] - feats_real[0]))
                loss_feat += wt * torch.mean(torch.abs(feats_fake[1] - feats_real[1]))
            for feats_fake, feats_real in zip(disc_fake_multiscale, disc_real_multiscale):
                loss_feat += wt * torch.mean(torch.abs(feats_fake[0] - feats_real[0]))
                loss_feat += wt * torch.mean(torch.abs(feats_fake[1] - feats_real[1]))

        adv_loss = 0.5 * adv_loss

        loss_g += hp.model.lambda_adv * adv_loss
        loss_g += hp.model.feat_match * loss_feat
        loss_d = loss_d_real + loss_d_fake
        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

        loader.set_description("g %.04f d %.04f ad %.04f| step %d" % (loss_g, loss_d, adv_loss, step))

    loss_g_avg = loss_g_sum / len(valloader.dataset)
    loss_d_avg = loss_d_sum / len(valloader.dataset)

    audio = audio[0][0].cpu().detach().numpy()
    fake_audio = fake_audio[0][0].cpu().detach().numpy()

    writer.log_validation(loss_g_avg, loss_d_avg, adv_loss, generator, discriminator, audio, fake_audio, step)

    torch.backends.cudnn.benchmark = True
    generator.train()
    discriminator.train()
