import tqdm
import torch
from plotting import get_files
from scipy.io.wavfile import write

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
        fake_audio = generator(mel) # B, 1, T' torch.Size([1, 1, 212992])

        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)]) # B, 1, T torch.Size([1, 1, 212893])
        disc_real = discriminator(audio)

        adv_loss =0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss


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
    
    mel_filename = get_files(hp.data.eval_path , extension = '.npy')
    for j in range(0,len(mel_filename)):
        mel = torch.from_numpy(np.load(mel_filename[j]))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        gen_audio = generator(mel)
        gen_audio = gen_audio.squeeze(0)
        gen_audio = gen_audio.squeeze()
        gen_audio = gen_audio[:-(hp.audio.hop_length*10)]
        gen_audio = MAX_WAV_VALUE * audio
        gen_audio = gen_audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        gen_audio = gen_audio.short()
        gen_audio = gen_audio.cpu().detach().numpy()
        out_path = mel_filename[j].replace('.npy', f'{step}.wav')
        mel_name = mel_filename.split("/")[-1].split(".")[0]
        writer.log_evaluation(gen_audio, step, mel_name)
        write(out_path, hp.audio.sampling_rate, gen_audio)
                   
    
    #add evalution code here

    torch.backends.cudnn.benchmark = True
    generator.train()
    discriminator.train()
