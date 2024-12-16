import torch
from torch.utils.data import Dataset
from .schema import *

class MultidimDataset(Dataset):
    def __init__(self, input_schema: DataSchema, output_schema: DataSchema, num_samples: int):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.num_samples = num_samples
        self.data = self._generate_data()

    def _generate_data(self):
        input_shape = (self.num_samples,) + self.input_schema.get_shape()
        output_shape = (self.num_samples,) + self.output_schema.get_shape()
        return {
            'input': torch.randn(input_shape),
            'output': torch.randn(output_shape)
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data['input'][idx], self.data['output'][idx]
