import time
from datetime import datetime
import os
import argparse
import torch
import numpy as np
import time
import torch.nn as nn
from dataloader import data_generator
from model import GaitGraphNet
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8,
                        help="seed")
    parser.add_argument("--device", type=int, default=0,
                        help="device")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--data_path", type=str, default='../data/PhysioNet',
                        help="data path")
    parser.add_argument("--label_file_path", type=str, default='../data/PhysioNet/demographics.csv',
                        help="label file path")
    parser.add_argument("--input_dim", type=int, default=100,
                        help="input dim")
    parser.add_argument("--output_dim", type=int, default=100,
                        help="output dim")
    parser.add_argument("--model_type", type=str, default='GCNConv',
                        help="model type ['GCNConv', 'GraphConv', 'GatedGraphConv']")
    parser.add_argument("--gcn_layers", type=int, default=2, help="gcn layers [2, 3, 4]")
    parser.add_argument("--attn_head", type=int, default=2, help="attn head [2, 4]")
    parser.add_argument("--encoder_layers", type=int, default=2,
                        help="attn head num [2, 4]")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="weight-decay")
    parser.add_argument('--save_dir', type=str, default='../result/',
                        help="save dir")
    args = parser.parse_args()

    # train_config
    seed = args.seed
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    data_path = args.data_path
    label_file_path = args.label_file_path
    input_dim = args.input_dim
    output_dim = args.output_dim
    model_type = args.model_type
    gcn_layers = args.gcn_layers
    attn_head = args.attn_head
    encoder_layers = args.encoder_layers
    epochs = args.epochs
    lr = args.lr
    save_dir = args.save_dir
    shuffle = True
    criterion = nn.CrossEntropyLoss()

    # 获取五折交叉验证的数据生成器
    train_loader, test_loader = data_generator(data_path, label_file_path, batch_size, shuffle, device, save_dir, seed)
    save_path = save_dir + f'/{seed}/{model_type}_gl_{gcn_layers}_ah_{attn_head}_el_{encoder_layers}'
    print(f"Random Seed: {seed}")
    print(f"Train batch size: {len(train_loader.dataset)}")
    print(f"Test batch size: {len(test_loader.dataset)}")
    model = GaitGraphNet(input_dim, output_dim, model_type, gcn_layers, attn_head, encoder_layers, 2).to(device)
    trainer = Trainer(model, epochs, batch_size, lr, train_loader, test_loader, criterion, device, save_path)
    trainer.train()
