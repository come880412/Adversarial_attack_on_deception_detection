import numpy as np
import math

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

lie_annotation = {1:'truth', 0:'lie',
                  'lie':0, 'truth':1}

def train_val_split(data_path, root_csv, dataset_name, train_ratio = 0.9, ):
    file_list = np.loadtxt(root_csv, delimiter=',', dtype=np.str)
    lie_name_list = []
    truth_name_list = []

    for idx, data in enumerate(file_list):
      folder_name = data[0]

      if dataset_name == 'Real_life':
        class_name = folder_name.split('_')[0]
        if class_name == 'lie':
          if folder_name not in lie_name_list:
            lie_name_list.append(folder_name)
        else:
          if folder_name not in truth_name_list:
            truth_name_list.append(folder_name)

      elif dataset_name == 'Bag_of_lies':
        class_name = folder_name.split('_')[1]
        if class_name == '0':
          if folder_name not in lie_name_list:
            lie_name_list.append(folder_name)
        else:
          if folder_name not in truth_name_list:
            truth_name_list.append(folder_name)
    lie_name_list = np.array(lie_name_list)
    truth_name_list = np.array(truth_name_list)

    train_name_list = []
    valid_name_list = []

    for name_list in [lie_name_list, truth_name_list]:
      len_dataname = len(name_list)
      random_num_list = np.random.choice(len_dataname, len_dataname, replace=False)

      train_index = np.array(random_num_list[:int(len(random_num_list) * train_ratio)], dtype=int)
      val_index = np.array(random_num_list[int(len(random_num_list) * train_ratio):], dtype=int)

      train_name_list += list(name_list[train_index])
      valid_name_list += list(name_list[val_index])
    np.savetxt('%s/%s_train.txt' % (data_path, dataset_name), train_name_list, fmt='%s')
    np.savetxt('%s/%s_val.txt' % (data_path, dataset_name), valid_name_list, fmt='%s')

    train_data = []
    valid_data = []
    for data in file_list:
      folder_name = data[0]
      if folder_name in train_name_list:
        train_data.append(data)
      elif folder_name in valid_name_list:
        valid_data.append(data)

    return train_data, valid_data


class Smoother(nn.Module):
  """Convolutional Transformer Encoder Layer"""

  def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout=0.1):
    super(Smoother, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    self.conv1 = nn.Conv1d(d_model, dim_feedforward, 5, padding=2)
    self.conv2 = nn.Conv1d(dim_feedforward, d_model, 1, padding=0)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(
    self,
    src,
    src_mask,
    src_key_padding_mask):
    # multi-head self attention
    src2 = self.self_attn(
        src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
    )[0]

    # add & norm
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    # conv1d
    src2 = src.transpose(0, 1).transpose(1, 2)
    src2 = self.conv2(F.relu(self.conv1(src2)))
    src2 = src2.transpose(1, 2).transpose(0, 1)

    # add & norm
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src

def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)