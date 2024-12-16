import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Datasets.schema import DataSchema

class SimpleLinearNet(nn.Module):
    def __init__(self, input_schema: DataSchema, output_schema: DataSchema):
        super(SimpleLinearNet, self).__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema

        input_size = np.prod(input_schema.get_shape())
        output_size = np.prod(output_schema.get_shape())

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view((-1,) + self.output_schema.get_shape())
