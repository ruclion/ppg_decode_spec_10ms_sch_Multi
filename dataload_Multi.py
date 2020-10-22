import os
import numpy as np
from torch.utils.data import Dataset



PPG_DIR =   '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/ppg_from_generate_batch'
MEL_DIR =  '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/mel_5ms_by_audio_2'
SPEC_DIR =  '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/spec_5ms_by_audio_2'
max_length = 1200
PPG_DIM = 345
MEL_DIM = 80
SPEC_DIM = 201



def text2list_DataBakerCN(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.strip().split('|')[0])
    print(file_list[:3])
    return file_list


def get_single_data_pair(fname, ppg_dir, mel_dir, spec_dir):
    ppg_f = os.path.join(ppg_dir, fname+'.npy')
    mel_f = os.path.join(mel_dir, fname+'.npy')
    spec_f = os.path.join(spec_dir, fname+'.npy')

    ppg = np.load(ppg_f)
    mel = np.load(mel_f)
    spec = np.load(spec_f)
    assert mel.shape[0] == ppg.shape[0] and mel.shape[0] == spec.shape[0], fname + ' 维度不相等'
    assert mel.shape[1] == MEL_DIM and ppg.shape[1] == PPG_DIM and spec.shape[1] == SPEC_DIM, fname + ' 特征维度不正确'
    return ppg, mel, spec


class DataBakerCNDataset(Dataset):
  def __init__(self, meta_list_path):
    self.file_list = text2list_DataBakerCN(file=meta_list_path)
    # 先延用长河的，所有batch的序列均padding为2000
    self.max_length = max_length

  # 不知道用处，可能是语法，先留着
  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self, idx):
    fname = self.file_list[idx]
    ppg, mel, spec = get_single_data_pair(fname, PPG_DIR, MEL_DIR, SPEC_DIR)
    ppg_len = ppg.shape[0]

    # if ppg_len > self.max_length:
    #     assert False



    # 此部分先没改
    pad_length = self.max_length - ppg.shape[0]
    if pad_length > 0:
      ppg_padded = np.vstack((ppg, np.zeros((pad_length, PPG_DIM))))
      mel_padded = np.vstack((mel, np.zeros((pad_length, MEL_DIM))))
      spec_padded = np.vstack((spec, np.zeros((pad_length, SPEC_DIM))))
    else:
      # print("BIGGER")
      ppg_padded = ppg[:self.max_length, :]
      mel_padded = mel[:self.max_length, :]
      spec_padded = spec[:self.max_length, :]
      ppg_len = self.max_length

    ppg_padded = ppg_padded.astype(np.float64)
    mel_padded = mel_padded.astype(np.float64)
    spec_padded = spec_padded.astype(np.float64)

    return ppg_padded, mel_padded, spec_padded, ppg_len