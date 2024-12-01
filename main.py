import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import List, Tuple, Dict

#*************Import Model Structure***************************************

from models import MultidimTransformerModel


'''
The core functionality of this code is to build a general framework for handling multi-dimensional data modeling of arbitrary dimensions, including:

- Dynamically defining data dimensions and model inputs/outputs.
- Generating simulated multi-dimensional data.
- Using a simple fully connected neural network for modeling.
- Providing a complete process for training and testing.

Users can easily adapt to different multi-dimensional data modeling tasks by adjusting the dimensions, network structure, and training parameters.
'''

'''
Modifications to the dimensions can be made by modifying the input_schema and output_schema.

    # Define input and output dimensions
    input_schema = DataSchema([
        #DataDimension("dim1", 24),
        #DataDimension("dim2", 5),
        DataDimension("dim3", 10),
        DataDimension("dim4", 10)
    ])

    output_schema = DataSchema([
        #DataDimension("dim1", 6),
        #DataDimension("dim2", 1),
        #DataDimension("dim3", 10),
        DataDimension("dim4", 10)
    ])
'''


class DataDimension:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size


class DataSchema:
    def __init__(self, dimensions: List[DataDimension]):
        self.dimensions = dimensions

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(dim.size for dim in self.dimensions)


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


class MultidimNet(nn.Module):
    def __init__(self, input_schema: DataSchema, output_schema: DataSchema):
        super(MultidimNet, self).__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema

        # 动态计算输入和输出大小
        input_size = np.prod(input_schema.get_shape())
        output_size = np.prod(output_schema.get_shape())

        # 简化的网络结构，可以根据需要进行调整
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view((-1,) + self.output_schema.get_shape())


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)  # 使用MSE损失
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='PyTorch Multidim Data Modeling')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'mamba2'],
                        help='Model type to use: transformer or mamba2')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 定义输入和输出维度
    input_schema = DataSchema([
        
        DataDimension("dim1", 5),
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
   
        
       


       
       
        
    ])

    output_schema = DataSchema([
       
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
  
        
        
      
        
    ])

    # 创建数据集
    train_dataset = MultidimDataset(input_schema, output_schema, num_samples=10000)
    test_dataset = MultidimDataset(input_schema, output_schema, num_samples=2000)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)


    '''
    ****************Set Model Structure**************************
    '''
    #model = MultidimNet(input_schema, output_schema).to(device)
    #model = MultidimTransformerModel(input_schema, output_schema).to(device)
    #model = NdMamba2(64, 128, 64).cuda()
    if args.model == 'transformer':
        model = MultidimTransformerModel(input_schema, output_schema).to(device)
    #elif args.model == 'mamba2':
    #    model = MultidimMamba2Model(input_schema, output_schema, device=device).to(device)
    
    

    '''
    ****************Set Model Structure**************************
    '''



    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "multidim_model.pt")


if __name__ == '__main__':
    main()
