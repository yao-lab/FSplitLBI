#  Copyright (C) 2019 Yanwei Fu, Chen Liu, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from slbi import SLBI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import alexnet
import numpy as np
from data_loader import load_data
import argparse
from torchvision import transforms
from utils import *
torch.manual_seed(24)
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--kappa", default=10,  type=int)
parser.add_argument("--interval", default=30, type=int)
parser.add_argument("--dataset", default='ImageNet', type=str)
parser.add_argument("--train", default=True, type=str2bool)
parser.add_argument("--download", default=False, type=str2bool)
parser.add_argument("--shuffle", default=True, type=str2bool)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=True, type=str2bool)
parser.add_argument("--epoch", default=150, type=int)
parser.add_argument("--gpu_num", default='0', type=str)
parser.add_argument("--mu", default=500, type=int)
parser.add_argument("--optimizer_type", default='slbi', type=str)
args = parser.parse_args()
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache()
model = alexnet.alexnet().to(device)
if args.parallel:
    model = nn.DataParallel(model)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
name_list = []
for name, p in model.named_parameters():
    name_list.append(name)
optimizer = SLBI(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu, weight_decay=0)
optimizer.assign_name(name_list)
train_loader = load_data(dataset=args.dataset, train=args.train, batch_size=args.batch_size, shuffle=args.shuffle)
test_loader = load_data(dataset=args.dataset, train=False, batch_size=64, shuffle=False)
all_num = args.epoch * len(train_loader)
print('num of all step:', all_num)
print('num of step per epoch:', len(train_loader))
for ep in range(args.epoch):
    model.train()
    descent_lr(args.lr, ep, optimizer, args.interval)
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        if (iter + 1) % 100 == 0:
            optimizer.step(closure=None, record=True, path='alexnet/')
        else:
            optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
        if (iter + 1) % 100 == 0:
            print('*******************************')
            print('epoch : ', ep + 1)
            print('iteration : ', iter + 1)
            print('loss : ', loss_val/100)
            print('Correct : ', correct)
            print('Num : ', num)
            print('Train ACC : ', correct/num)
            optimizer.calculate_difference()
            correct = num = 0
            loss_val = 0
    print('Test Model')
    evaluate_batch(model, test_loader, device)
