import os
import numpy as np
from torch.utils.data import Dataset



English_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG-old/ppg_generate_10ms_by_audio_hjk2'
English_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG-old/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG-old/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG-old/ppg_generate_10ms_by_audio_hjk2'
MEL_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG-old/mel_10ms_by_audio_hjk2'
MEL_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG-old/mel_10ms_by_audio_hjk2'
SPEC_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG-old/spec_10ms_by_audio_hjk2'
SPEC_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG-old/spec_10ms_by_audio_hjk2'
max_length = 1000
PPG_DIM = 347 + 218
MEL_DIM = 80
SPEC_DIM = 201



def text2list_Multi(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.strip())
    return file_list


def get_single_data_pair(fname, speaker_id, english_ppg_dir, mandarin_ppg_dir, mel_dir, spec_dir):
    english_ppg_f = os.path.join(english_ppg_dir, fname+'.npy')
    mandarin_ppg_f = os.path.join(mandarin_ppg_dir, fname+'.npy')
    mel_f = os.path.join(mel_dir, fname+'.npy')
    spec_f = os.path.join(spec_dir, fname+'.npy')

    english_ppg = np.load(english_ppg_f)
    mandarin_ppg = np.load(mandarin_ppg_f)
    bilingual_ppg = np.concatenate((english_ppg, mandarin_ppg),axis=-1)
    # print('bilingual shape:', bilingual_ppg.shape)
    mel = np.load(mel_f)
    spec = np.load(spec_f)
    assert mel.shape[0] == bilingual_ppg.shape[0] and mel.shape[0] == spec.shape[0], fname + ' 维度不相等'
    assert mel.shape[1] == MEL_DIM and bilingual_ppg.shape[1] == PPG_DIM and spec.shape[1] == SPEC_DIM, fname + ' 特征维度不正确'
    return bilingual_ppg, mel, spec


class MultiDataset(Dataset):
  def __init__(self, meta_list_path):
    self.file_list = text2list_Multi(file=meta_list_path)
    # 先延用长河的，所有batch的序列均padding为2000
    self.max_length = max_length

  # 不知道用处，可能是语法，先留着
  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self, idx):
    tmp = self.file_list[idx].split('|')
    fname = tmp[0]
    speaker_id = int(tmp[1])
    # LJSpeech
    if speaker_id == 0:
      ppg, mel, spec = get_single_data_pair(fname, speaker_id, English_PPG_LJSpeech_DIR, Mandarin_PPG_LJSpeech_DIR, MEL_LJSpeech_DIR, SPEC_LJSpeech_DIR)
    # DataBaker
    if speaker_id == 1:
      ppg, mel, spec = get_single_data_pair(fname, speaker_id, English_PPG_DataBaker_DIR, Mandarin_PPG_DataBaker_DIR, MEL_DataBaker_DIR, SPEC_DataBaker_DIR)
    
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

    return ppg_padded, mel_padded, spec_padded, ppg_len, speaker_id