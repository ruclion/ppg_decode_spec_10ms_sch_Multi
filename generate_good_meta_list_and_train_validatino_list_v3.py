import os
import numpy as np
from tqdm import tqdm


np.random.seed(2117)
validation_size = 2000


# in
English_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/ppg_generate_10ms_by_audio_hjk2'
English_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG/ppg_generate_10ms_by_audio_hjk2'
Mandarin_PPG_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG/ppg_generate_10ms_by_audio_hjk2'
english_ppg_MEL_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/mel_10ms_by_audio_hjk2'
mandarin_ppg_MEL_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG/mel_10ms_by_audio_hjk2'
english_ppg_MEL_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/mel_10ms_by_audio_hjk2'
mandarin_ppg_MEL_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG/mel_10ms_by_audio_hjk2'
english_ppg_SPEC_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/spec_10ms_by_audio_hjk2'
mandarin_ppg_SPEC_LJSpeech_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG/spec_10ms_by_audio_hjk2'
english_ppg_SPEC_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/spec_10ms_by_audio_hjk2'
mandarin_ppg_SPEC_DataBaker_DIR = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG/spec_10ms_by_audio_hjk2'

# out
English_PPG_LJSpeech_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG/meta_good_fileList_v3.txt'
English_PPG_DataBaker_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG/meta_good_fileList_v3.txt'
Mandarin_PPG_LJSpeech_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG/meta_good_fileList_v3.txt'
Mandarin_PPG_DataBaker_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG/meta_good_fileList_v3.txt'

Merge_PPG_LJSpeech_meta_path = './meta_good_merge_ljspeech_v3.txt'
Merge_PPG_DataBaker_meta_path = './meta_good_merge_databaker_v3.txt'

train_file_path = './meta_good_v3_train.txt'
validation_file_path = './meta_good_v3_validation.txt'



def dirPath2list(dir_path, ppg_dir_path, mel_dir_path, spec_dir_path):
    print('start...', dir_path)
    fname_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    ans = []
    t = 0
    for i in tqdm(fname_files):
        try:
            ppg_path = os.path.join(ppg_dir_path, i)
            mel_path = os.path.join(mel_dir_path, i)
            spec_path = os.path.join(spec_dir_path, i)
            a = np.load(ppg_path)
            b = np.load(mel_path)
            c = np.load(spec_path)
            assert a.shape[0] == b.shape[0] and b.shape[0] == c.shape[0]
            ans.append(i.split('.')[0])
        except Exception as e:
            print(i, 'is bad')
            print(str(e))
        t += 1
        # if t > 5:
        #     break
    print('end...')
    return ans


def list2file(lst, file):
    with open(file, 'w') as f:
        for i in lst:
            f.write(i + '\n')


def mergeList(l1, l2):
    # 1e4级别, 不用管复杂度

    # print(l1)
    # print(l2)
    print('start merge...')
    ans = []
    for i in tqdm(l1):
        if i in l2:
            ans.append(i)
    print('end...')
    return ans


def main():
    # 保证文件不损坏
    English_PPG_LJSpeech_meta_list = dirPath2list(English_PPG_LJSpeech_DIR, English_PPG_LJSpeech_DIR, english_ppg_MEL_LJSpeech_DIR, english_ppg_SPEC_LJSpeech_DIR)
    English_PPG_DataBaker_meta_list = dirPath2list(English_PPG_DataBaker_DIR, English_PPG_DataBaker_DIR, english_ppg_MEL_DataBaker_DIR, english_ppg_SPEC_DataBaker_DIR)
    Mandarin_PPG_LJSpeech_meta_list = dirPath2list(Mandarin_PPG_LJSpeech_DIR, Mandarin_PPG_LJSpeech_DIR, mandarin_ppg_MEL_LJSpeech_DIR, mandarin_ppg_SPEC_LJSpeech_DIR)
    Mandarin_PPG_DataBaker_meta_list = dirPath2list(Mandarin_PPG_DataBaker_DIR, Mandarin_PPG_DataBaker_DIR, mandarin_ppg_MEL_DataBaker_DIR, mandarin_ppg_SPEC_DataBaker_DIR)
    
    list2file(English_PPG_LJSpeech_meta_list, English_PPG_LJSpeech_meta_path)
    list2file(English_PPG_DataBaker_meta_list, English_PPG_DataBaker_meta_path)
    list2file(Mandarin_PPG_LJSpeech_meta_list, Mandarin_PPG_LJSpeech_meta_path)
    list2file(Mandarin_PPG_DataBaker_meta_list, Mandarin_PPG_DataBaker_meta_path)

    # 保证中英PPG都有
    Merge_PPG_LJSpeech_meta_list = mergeList(English_PPG_LJSpeech_meta_list, Mandarin_PPG_LJSpeech_meta_list)
    Merge_PPG_DataBaker_meta_list = mergeList(English_PPG_DataBaker_meta_list, Mandarin_PPG_DataBaker_meta_list)
    
    list2file(Merge_PPG_LJSpeech_meta_list, Merge_PPG_LJSpeech_meta_path)
    list2file(Merge_PPG_DataBaker_meta_list, Merge_PPG_DataBaker_meta_path)

    # 切分
    all_list = []
    for i in Merge_PPG_LJSpeech_meta_list:
        all_list.append(i + '|' + '0')
    for i in Merge_PPG_DataBaker_meta_list:
        all_list.append(i + '|' + '1')
    # print(all_list)
    len_list = len(all_list)
    small_validation_size = np.minimum(validation_size, int(0.1 * len_list))
    print('validation:', small_validation_size)

    np.random.shuffle(all_list)
    validation_list = all_list[:small_validation_size]  
    train_list = all_list[small_validation_size:]

    list2file(train_list, train_file_path)
    list2file(validation_list, validation_file_path)


if __name__ == '__main__':
    main()