# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:40:15 2020

@author: gliu3
"""

import torch
lstm = torch.nn.LSTM(input_size=5, hidden_size=3, bidirectional=True)
seq_len, batch, input_size, num_directions = 3, 1, 5, 2
in_data = torch.randint(10, (seq_len, batch, input_size)).float()
output, (h_n, c_n) = lstm(in_data)

print(output)

print(h_n)