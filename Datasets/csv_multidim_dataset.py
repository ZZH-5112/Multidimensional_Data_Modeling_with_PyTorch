import csv
import torch
from torch.utils.data import Dataset
from .schema import *

class CSVMultidimDataset(Dataset):
    def __init__(self, file_path: str, input_schema: DataSchema, output_schema: DataSchema):
        self.file_path = file_path
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.data = self._generate_data()
        self.num_samples = self.data['input'].shape[0]

    def _generate_data(self):
        input_data = self._read_csv_data()
        output_shape = (input_data.shape[0],) + self.output_schema.get_shape()
        return {
            'input': input_data,
            'output': torch.randn(output_shape)
        }

    def _read_csv_data(self):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            raw_data = [float(row[2]) for row in reader]

        input_shape = self.input_schema.get_shape()
        elements_per_sample = torch.prod(torch.tensor(input_shape)).item()
        num_complete_samples = len(raw_data) // elements_per_sample

        if num_complete_samples == 0:
            raise ValueError(f"Not enough data in CSV file to create even one complete sample. Expected at least {elements_per_sample} elements.")

        usable_data = raw_data[:num_complete_samples * elements_per_sample]
        return torch.tensor(usable_data).reshape(num_complete_samples, *input_shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data['input'][idx], self.data['output'][idx]

