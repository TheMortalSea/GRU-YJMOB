import os
import argparse
import logging
import random
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# import wandb

from dataset import *
from model import *


path_arr = [
    './dataset/city_A_challengedata.csv.gz',
    './dataset/city_B_challengedata.csv.gz',
    './dataset/city_C_challengedata.csv.gz',
    './dataset/city_D_challengedata.csv.gz'
]

# 设置随机种子以确保结果的可重复性
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 将一个批次的数据样本整合成一个张量
def collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    city = [item['city'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    len_tensor = torch.tensor([item['len'] for item in batch])

    # 将样本填充至相同长度，填充值均为0
    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    city_padded = pad_sequence(city, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)

    # 返回字典，包含填充后的张量
    return {
        'd': d_padded,
        't': t_padded,
        'input_x': input_x_padded,
        'input_y': input_y_padded,
        'time_delta': time_delta_padded,
        'city': city_padded,
        'label_x': label_x_padded,
        'label_y': label_y_padded,
        'len': len_tensor
    }

# 微调函数
def finetune(args):

    # 设置日志文件名
    # name = f'finetune_batchsize{args.batch_size}_epochs{args.epochs}_embedsize{args.embed_size}_layersnum{args.layers_num}_headsnum{args.heads_num}_cuda{args.cuda}_lr{args.lr}_seed{args.seed}'
    name = 'LPBERT-postembedAB-finetuneC'
    current_time = datetime.datetime.now()
    # 初始化 wandb
    # wandb.init(project="LPBERT", name=name, config=args)
    # wandb.run.name = name  # Set the run name
    # wandb.run.save()

    # 加载训练集
    dataset_train = TrainSet(path_arr[2])
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    # 通过cuda:<device_id>指定使用的GPU
    device = torch.device(f'cuda:{args.cuda}')

    # 实例化LP-BERT模型，并加载至GPU上
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size, args.cityembed_size).to(device)
    model.load_state_dict(torch.load(args.pretrained_model))
    
    # 冻结部分参数
    for name, param in model.named_parameters():
        if 'transformer_encoder.transformer_encoder.layers.3' in name or 'ffn_layer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train()

    # 指定Adam优化器、CosineAnnealingLR学习率调度器、交叉熵损失函数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 微调循环
    for epoch_id in range(args.epochs):
        total_epoch_loss = 0
        for batch_id, batch in enumerate(tqdm(dataloader_train)):

            # 按批次将数据加载至GPU中
            batch['d'] = batch['d'].to(device)
            batch['t'] = batch['t'].to(device)
            batch['input_x'] = batch['input_x'].to(device)
            batch['input_y'] = batch['input_y'].to(device)
            batch['time_delta'] = batch['time_delta'].to(device)
            batch['city'] = batch['city'].to(device)
            batch['label_x'] = batch['label_x'].to(device)
            batch['label_y'] = batch['label_y'].to(device)
            batch['len'] = batch['len'].to(device)

            # 将数据输入模型中得到输出
            output = model(batch['d'], batch['t'], batch['input_x'], batch['input_y'], batch['time_delta'], batch['len'], batch['city'])

            # 将x和y堆叠成一个张量
            label = torch.stack((batch['label_x'], batch['label_y']), dim=-1)

            # 创建预测掩码，并将其扩展至与label相同的维度
            pred_mask = (batch['input_x'] == 201)
            pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)

            # 计算损失，反向传播计算梯度并更新模型参数，清除累积梯度
            loss = criterion(output[pred_mask], label[pred_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_epoch_loss += loss.detach().item()

            step = epoch_id * len(dataloader_train) + batch_id
        
        # After the inner loop, calculate the average loss for the epoch
        avg_epoch_loss = total_epoch_loss / len(dataloader_train)
        
        # Log the average epoch loss
        # wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch_id})

        scheduler.step()

        best_loss = float('inf')

        # Now, use the average epoch loss for the saving condition
        current_time = datetime.datetime.now()
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_dir = '/content/drive/MyDrive'
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = f'{save_dir}/best_finetune_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch_id + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f} - NEW BEST! Model saved to {model_save_path}")
        else:
            print(f"Epoch {epoch_id + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f} - Best loss still: {best_loss:.4f}")




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='/content/drive/MyDrive/best_pretrain_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)  # 微调可以选择较少的epochs
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--cityembed_size', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=3704)
    args = parser.parse_args()

    set_random_seed(args.seed)

    finetune(args)