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
from __future__ import division, print_function
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
'''
This file is used to load data
Data used in this project includes MNIST, Cifar10 and ImageNet
'''


def load_data(dataset='Cifar10', train=True, download=True, transform=None, batch_size=1, shuffle=True):
	if dataset == 'MNIST':
		data_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=train, download=download, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=batch_size, shuffle=shuffle)
	elif dataset == 'Cifar10':
		if train:
			trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
		else:
			trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
		data_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data/Cifar10', train=train, download=download, transform=trans), batch_size=batch_size, shuffle=shuffle)

	elif dataset == 'ImageNet':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		if train:
			dataset = torchvision.datasets.ImageFolder('/home/wwx/ad_attack/Imagenet/ILSVRC2012/train/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)
		else:
			dataset = torchvision.datasets.ImageFolder('/home/wwx/ad_attack/Imagenet/ILSVRC2012/val/', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)
	else:
		print('No such dataset')
	return data_loader


