#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:16:41 2020

@author: Xuye Liu
"""
from models.HAConvGNN import HAConvGNN
import torch


def create_model(config):
    mdl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = HAConvGNN(config, device).to(device)
    print(torch.cuda.device_count())
    return mdl, device
