import os
import numpy as np


np.random.seed(2117)
file_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good.txt'
train_file_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good_train.txt'
validation_file_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good_validation.txt'
validation_size = 1000


def text2list_ljspeech(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.strip().split('|')[0])
    return file_list


def list2test_fname(lst, file):
    with open(file, 'w') as f:
        for i in lst:
            f.write(i + '\n')



all_list = text2list_ljspeech(file=file_path)
len_list = len(all_list)

np.random.shuffle(all_list)
train_list = all_list[:len_list - 1000]
validation_list = all_list[len_list - 1000:]

list2test_fname(train_list, train_file_path)
list2test_fname(validation_list, validation_file_path)



