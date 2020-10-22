# 不会的问题：GriffinLim的power和iter不会设置，也不知道有什么影响，长河的1.0和100，常用的是1.5和60
import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model_torch import DCBHG
from dataload_DataBakerCN import DataBakerCNDataset
from audio import hparams as audio_hparams
from audio import normalized_db_mel2wav, normalized_db_spec2wav, write_wav


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 80,
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

TRAIN_FILE = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good_train.txt'
VALIDATION_FILE = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good_validation.txt'
# 注意是否要借鉴已经有的模型
restore_ckpt_path_DataBakerCN = '/datapool/home/hujk17/ppg_decode_spec_5ms_sch_DataBakerCN/const_ckpt/checkpoint_step000034800.pth'


use_cuda = torch.cuda.is_available()
assert use_cuda is True
device = torch.device("cuda" if use_cuda else "cpu")
num_workers = 0

# some super parameters，用epochs来计数，而不是步数（不过两者同时统计）
# BATCH_SIZE = 64
# BATCH_SIZE = 2
BATCH_SIZE = 16
clip_thresh = 0.1
nepochs = 5000
LEARNING_RATE = 0.0003
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
CKPT_EVERY = 300
# CKPT_EVERY = 2
VALIDATION_EVERY = 600
# VALIDATION_EVERY = 2

# DataBakerCN的log: ckpt文件夹以及wav文件夹，tensorboad在wav文件夹中
DataBakerCN_log_dir = os.path.join('restoreANDvalitation_DataBakerCN_log_dir', STARTED_DATESTRING, 'train_wav')
DataBakerCN_model_dir = os.path.join('restoreANDvalitation_DataBakerCN_log_dir', STARTED_DATESTRING, 'ckpt_model')
if os.path.exists(DataBakerCN_log_dir) is False:
  os.makedirs(DataBakerCN_log_dir, exist_ok=True)
if os.path.exists(DataBakerCN_model_dir) is False:
  os.makedirs(DataBakerCN_model_dir, exist_ok=True)



# 恢复训练，需要测试对不对，特别是loss是不是接着下降的
def load_checkpoint(checkpoint_path, model, optimizer):
  assert os.path.isfile(checkpoint_path)
  print("Loading checkpoint '{}'".format(checkpoint_path))

  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model.load_state_dict(checkpoint_dict['state_dict'])
  optimizer.load_state_dict(checkpoint_dict['optimizer'])
  global_step = checkpoint_dict['global_step']
  global_epoch = checkpoint_dict['global_epoch']
  print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, global_step))
  return model, optimizer, global_step, global_epoch


def validate(model, criterion, validation_torch_loader, now_steps, writer):
  print('Start validation...')
  model.eval()
  with torch.no_grad():
    val_loss = 0.0
    for _step, (ppgs, mels, specs, lengths) in tqdm(enumerate(validation_torch_loader)):
      # 数据拿到GPU上
      ppgs = ppgs.to(device)
      mels = mels.to(device)
      specs = specs.to(device)
      ppgs, mels, specs = Variable(ppgs).float(), Variable(mels).float(), Variable(specs).float()
      # Batch同时计算出pred结果
      mels_pred, specs_pred = model(ppgs)
      # 根据预测结果定义/计算loss
      loss = 0.0
      # print('now validation batch size:', lengths.shape[0])
      for i in range(lengths.shape[0]):
        mel_loss = criterion(mels_pred[i, :lengths[i], :], mels[i, :lengths[i], :])
        spec_loss = criterion(specs_pred[i, :lengths[i], :], specs[i, :lengths[i], :])
        loss += (mel_loss + spec_loss)

      loss = loss / BATCH_SIZE
      val_loss += loss.item() 

    # 计算验证集整体loss，然后画出来
    val_loss /= (len(validation_torch_loader))
    # writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
    writer.add_scalar("validation_loss", val_loss, now_steps)
    # 合成声音
    id = 0
    generate_pair_wav(specs[id, :lengths[id], :].cpu().data.numpy(), specs_pred[id, :lengths[id], :].cpu().data.numpy(), DataBakerCN_log_dir, now_steps, suffix_name='first')
    id = lengths.shape[0] - 1
    generate_pair_wav(specs[id, :lengths[id], :].cpu().data.numpy(), specs_pred[id, :lengths[id], :].cpu().data.numpy(), DataBakerCN_log_dir, now_steps, suffix_name='last')

  model.train()
  print('ValidationLoss:', val_loss)


def generate_pair_wav(spec, spec_pred, log_dir, global_step, suffix_name):
  y_pred = normalized_db_spec2wav(spec_pred)
  pred_wav_path = os.path.join(log_dir, "step_" + str(global_step) + "_" + suffix_name + "_predvalidation.wav")
  write_wav(pred_wav_path, y_pred)
  pred_spec_path = os.path.join(log_dir, "step_" + str(global_step) + "_" + suffix_name + "_predvalidation.npy")
  np.save(pred_spec_path, spec_pred)


  y = normalized_db_spec2wav(spec)
  orig_wav_path = os.path.join(log_dir, "step_" + str(global_step) + "_" + suffix_name + "_original.wav")
  write_wav(orig_wav_path, y)
  orig_spec_path = os.path.join(log_dir, "step_" + str(global_step) + "_" + suffix_name + "_original.npy")
  np.save(orig_spec_path, spec)


def main():
  # 数据读入，准备
  now_dataset_train = DataBakerCNDataset(TRAIN_FILE)
  now_train_torch_dataloader = DataLoader(now_dataset_train, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last=True)

  now_dataset_validation = DataBakerCNDataset(VALIDATION_FILE)
  now_validation_torch_loader = DataLoader(now_dataset_validation, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True)
  
  
  # 构建模型，放在gpu上，顺便把tensorboard的图的记录变量操作也算在这里面
  model = DCBHG().to(device)
  writer = SummaryWriter(log_dir=DataBakerCN_log_dir)


  # 设置梯度回传优化器，目前使用固定lr=0.0003，不知用不用变lr
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


  global_step = 0
  global_epoch = 0
  if restore_ckpt_path_DataBakerCN is not None:
    model, optimizer, _step, _epoch = load_checkpoint(restore_ckpt_path_DataBakerCN, model, optimizer)
    global_step = _step
    global_epoch = _epoch
  

  # optimize classification
  # cross_entropy_loss = nn.CrossEntropyLoss()
  # criterion = nn.MSELoss()
  # l1_loss = nn.NLLLoss()
  # from kuaishou 
  my_l1_loss = nn.L1Loss()


  


  # 开始训练
  print('Start Training...')
  
  model.train()
  while global_epoch < nepochs:
      running_loss = 0.0
      for _step, (ppgs, mels, specs, lengths) in tqdm(enumerate(now_train_torch_dataloader)):
          # Batch开始训练，清空opt，数据拿到GPU上
          optimizer.zero_grad()

          ppgs = ppgs.to(device)
          mels = mels.to(device)
          specs = specs.to(device)
          ppgs, mels, specs = Variable(ppgs).float(), Variable(mels).float(), Variable(specs).float()


          # Batch同时计算出pred结果
          mels_pred, specs_pred = model(ppgs)


          # 根据预测结果定义/计算loss; 不过我记得tacotron里面不是用的两个l1loss吧，之后再看看 TODO
          loss = 0.0
          for i in range(BATCH_SIZE):
            mel_loss = my_l1_loss(mels_pred[i, :lengths[i], :], mels[i, :lengths[i], :])
            spec_loss = my_l1_loss(specs_pred[i, :lengths[i], :], specs[i, :lengths[i], :])
            loss += (mel_loss + spec_loss)
          loss = loss / BATCH_SIZE
          print('Steps', global_step, 'Training Loss：', loss.item())
          writer.add_scalar("loss", float(loss.item()), global_step)
          running_loss += loss.item() 


          # 根据loss，计算梯度，并且应用梯度回传操作，改变权重值
          loss.backward()
          if clip_thresh > 0:
            _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh) # 返回值不用管
          optimizer.step()


          # 存储ckpt，存储生成的音频
          if global_step > 0 and global_step % CKPT_EVERY == 0:
            checkpoint_path = os.path.join(DataBakerCN_model_dir, "checkpoint_step{:09d}.pth".format(global_step))
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "global_epoch": global_epoch,
            }, checkpoint_path)
            
          # 测试集的效果，很重道，不过均在这一个函数中实现了
          if global_step > 0 and global_step % VALIDATION_EVERY == 0:
            validate(model=model, criterion=my_l1_loss, validation_torch_loader=now_validation_torch_loader, now_steps=global_step, writer=writer)

          # 该BATCH操作结束，step++
          global_step += 1

      # 对整个epoch进行信息统计
      averaged_loss = running_loss / (len(now_train_torch_dataloader))
      writer.add_scalar('epochLoss', averaged_loss, global_epoch)
      global_epoch += 1


if __name__ == '__main__':
  main()
