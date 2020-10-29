import os
import numpy as np


np.random.seed(2117)
validation_size = 1000

# in
English_PPG_LJSpeech_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/LJSpeech-1.1-English-PPG-old/meta_good_small.txt'
English_PPG_DataBaker_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_librispeech/DataBaker-English-PPG-old/meta_good_small.txt'
Mandarin_PPG_LJSpeech_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/LJSpeech-1.1-Mandarin-PPG-old/meta_good_small.txt'
Mandarin_PPG_DataBaker_meta_path = '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/DataBaker-Mandarin-PPG-old/meta_good_small.txt'

# out
train_file_path = './meta_good_old_small_train.txt'
validation_file_path = './meta_good_old_small_validation.txt'



def text2list(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.strip())
    return file_list


def list2test_fname(lst, file):
    with open(file, 'w') as f:
        for i in lst:
            f.write(i + '\n')


def mergeList(l1, l2):
    # 1e4级别, 不用管复杂度
    ans = []
    for i in l1:
        if i in l2:
            ans.append(i)
    return ans


def main():
    ljspeech_list_en = text2list(file=English_PPG_LJSpeech_meta_path)
    ljspeech_list_cn = text2list(file=Mandarin_PPG_LJSpeech_meta_path)
    databaker_list_en = text2list(file=English_PPG_DataBaker_meta_path)
    databaker_list_cn = text2list(file=Mandarin_PPG_DataBaker_meta_path)
    # print(ljspeech_list_cn)
    ljspeech_list = mergeList(ljspeech_list_en, ljspeech_list_cn)
    databaker_list = mergeList(databaker_list_en, databaker_list_cn)

    all_list = []
    for i in ljspeech_list:
        all_list.append(i + '|' + '0')
    for i in databaker_list:
        all_list.append(i + '|' + '1')
    len_list = len(all_list)
    small_validation_size = np.minimum(validation_size, int(0.2 * len_list))

    np.random.shuffle(all_list)
    validation_list = all_list[:small_validation_size]  
    train_list = all_list[small_validation_size:]

    list2test_fname(train_list, train_file_path)
    list2test_fname(validation_list, validation_file_path)


if __name__ == '__main__':
    main()



