import tqdm
import torch


def validate(hp, args, generator, discriminator, valloader, stft_loss, sub_stft_loss, criterion, pqmf, writer, step):
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
        fake_audio = generator(mel) # B, 1, T' torch.Size([1, 1, 212992])
        if hp.model.out_channels > 1:
            y_mb_ = fake_audio
            fake_audio = pqmf.synthesis(fake_audio)
        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)]) # B, 1, T torch.Size([1, 1, 212893])
        disc_real = discriminator(audio)

        adv_loss =0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss

        if hp.model.use_subband_stft_loss:
            loss_g *= 0.5  # for balancing with subband stft loss
            y_mb = pqmf.analysis(audio)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = sub_stft_loss(y_mb_[:, :y_mb.size(-1)], y_mb)
            loss_g += 0.5 * (sub_sc_loss + sub_mag_loss)

        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            adv_loss += criterion(score_fake, torch.ones_like(score_fake))

            if hp.model.feat_loss :
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d_real += criterion(score_real, torch.ones_like(score_real))
            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
        adv_loss = adv_loss / len(disc_fake)
        loss_d_real = loss_d_real / len(score_real)
        loss_d_fake = loss_d_fake / len(disc_fake)
        loss_g += hp.model.lambda_adv * adv_loss
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
