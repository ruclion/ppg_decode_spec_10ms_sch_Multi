import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import torch
from torch.autograd import Variable

from audio import hparams as audio_hparams
from audio import normalized_db_mel2wav, normalized_db_spec2wav, write_wav

from model_torch import DCBHG


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': True,
}


assert hparams == audio_hparams

EN_PPG_DIM = 347
CN_PPG_DIM = 218
PPG_DIM = 347 + 218


use_cuda = torch.cuda.is_available()
assert use_cuda is True


# 超参数和路径
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
ckpt_path_Multi = '/datapool/home/hujk17/ppg_decode_spec_10ms_sch_Multi/Multi_v2_log_dir/2020-10-29T13-52-53/ckpt_model/checkpoint_step000043200.pth'

ppgs_paths = 'inference_ppgs_path_list.txt'
DataBakerCN_log_dir = os.path.join('inference_Multi_log_dir', STARTED_DATESTRING)
if os.path.exists(DataBakerCN_log_dir) is False:
    os.makedirs(DataBakerCN_log_dir, exist_ok=True)

# 全局变量
model = DCBHG()

def tts_load(model, ckpt_path):
    ckpt_load = torch.load(ckpt_path)
    model.load_state_dict(ckpt_load["state_dict"])
    if use_cuda:
        model = model.cuda()
    model.eval()
    return model


def tts_predict(model, ppg, id_speaker):
    # 准备输入的数据并转换到GPU
    ppg = Variable(torch.from_numpy(ppg)).unsqueeze(0).float()
    id_speaker = torch.LongTensor([id_speaker])
    print('orig:', id_speaker)
    print(id_speaker.shape)
    # id_speaker = id_speaker.unsqueeze(0)
    print(ppg.size())
    print(ppg.shape)
    print(ppg.type())
    print('---------- id_speaker')
    print(id_speaker.size())
    print(id_speaker.shape)
    print(id_speaker.type())
    print(id_speaker)
    if use_cuda:
        ppg = ppg.cuda()
        id_speaker = id_speaker.cuda()

    # 进行预测并数据转换到CPU
    mel_pred, spec_pred = model(ppg, id_speaker)
    mel_pred = mel_pred[0].cpu().data.numpy()
    spec_pred = spec_pred[0].cpu().data.numpy()

    # vocoder合成音频波形文件
    mel_pred_audio = normalized_db_mel2wav(mel_pred)
    spec_pred_audio = normalized_db_spec2wav(spec_pred)

    return mel_pred, spec_pred, mel_pred_audio, spec_pred_audio


def draw_spec(a_path, a):
    plt.imshow(a.T, cmap='hot', interpolation='nearest')
    plt.xlabel('frame nums')
    plt.ylabel('spec')
    plt.tight_layout()
    plt.savefig(a_path, format='png')


def main():
    global model
    model = tts_load(model=model, ckpt_path=ckpt_path_Multi)


    ppgs_list = open(ppgs_paths, 'r')
    ppgs_list = [i.strip() for i in ppgs_list]
    for idx, two_ppg_paths_speaker in tqdm(enumerate(ppgs_list)):
        en_ppg_path, cn_ppg_path, speaker_id = two_ppg_paths_speaker.split('|')
        en_ppg = np.load(en_ppg_path)
        cn_ppg = np.load(cn_ppg_path)
        assert en_ppg.shape[1] == EN_PPG_DIM and cn_ppg.shape[1] == CN_PPG_DIM
        bilingual_ppg = np.concatenate((en_ppg, cn_ppg),axis=-1)
        assert bilingual_ppg.shape[1] == PPG_DIM
        speaker_id = int(speaker_id)
        mel_pred, spec_pred, mel_pred_audio, spec_pred_audio = tts_predict(model, bilingual_ppg, speaker_id)

        write_wav(os.path.join(DataBakerCN_log_dir, "{}_sample_mel.wav".format(idx)), mel_pred_audio)
        write_wav(os.path.join(DataBakerCN_log_dir, "{}_sample_spec.wav".format(idx)), spec_pred_audio)

        np.save(os.path.join(DataBakerCN_log_dir, "{}_sample_mel.npy".format(idx)), mel_pred)
        np.save(os.path.join(DataBakerCN_log_dir, "{}_sample_spec.npy".format(idx)), spec_pred)

        draw_spec(os.path.join(DataBakerCN_log_dir, "{}_sample_mel.png".format(idx)), mel_pred)
        draw_spec(os.path.join(DataBakerCN_log_dir, "{}_sample_spec.png".format(idx)), spec_pred)
      
    

if __name__ == "__main__":
    main()