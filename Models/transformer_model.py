# transformer_model.py

import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class MultidimTransformerModel(nn.Module):
    def __init__(self, input_schema, output_schema, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(MultidimTransformerModel, self).__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema

        input_dim = np.prod(input_schema.get_shape())
        output_dim = np.prod(output_schema.get_shape())

        self.flatten = nn.Flatten()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.linear_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, dim1, dim2, dim3, ...)
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        x = self.flatten(x)  # (batch_size, input_dim)
        x = self.linear_in(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (1, batch_size, d_model)
        x = self.transformer_encoder(x)  # (1, batch_size, d_model)
        x = x.transpose(0, 1).squeeze(1)  # (batch_size, d_model)
        x = self.linear_out(x)  # (batch_size, output_dim)
        return x.view((-1,) + self.output_schema.get_shape())
