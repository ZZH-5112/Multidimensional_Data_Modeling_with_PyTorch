import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import List, Tuple, Dict


from Datasets.schema import DataDimension,DataSchema
from Models import MultidimTransformerModel,SimpleLinearNet
from Datasets import MultidimDataset,CSVMultidimDataset

'''
This code provides a flexible framework for modeling multi-dimensional data using PyTorch. It is designed to handle data with arbitrary dimensions and allows dynamic definition of input and output schemas. The core functionality includes:

- **Customizable Data Dimensions**: Users can specify input and output schemas by defining the dimensions of the data dynamically.
- **Support for Simulated and CSV-based Data**: Data can be generated randomly or loaded from CSV files for training and testing.
- **Flexible Model Selection**: Two types of models are supported: a simple fully connected neural network and a transformer-based model.
- **Training and Testing Framework**: Includes configurable hyperparameters, training, testing, and logging capabilities.
- **Extensible for Different Tasks**: The framework is easily adaptable to various tasks by changing the input/output schemas, network structure, and data source.

### Example Configuration
The current example uses:
- **Input Schema**: One dimension with 490 features.
- **Output Schema**: Three dimensions, with sizes 1, 10, and 10.
- The default model is a transformer-based model trained on random data.

This modular structure allows users to efficiently design and experiment with models tailored to their specific multi-dimensional data requirements.
'''

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
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'linear'],
                        help='Model type to use: transformer or simple linear')
    parser.add_argument('--dataset', type=str, default='random', choices=['random', 'csv'],
                        help='Dataset type to use: random or csv')
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

    #*DATASETS
    if args.dataset == 'random':
        train_dataset = MultidimDataset(input_schema,output_schema,num_samples=10000)
        test_dataset = MultidimDataset(input_schema,output_schema,num_samples=1000)
    elif args.dataset == 'csv':
        train_dataset = CSVMultidimDataset('Data/sample.csv',input_schema,output_schema)
        test_dataset = CSVMultidimDataset('Data/sample.csv',input_schema,output_schema)
    
    #*DATALOADER
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    #*MODELS
    if args.model == 'linear':
        model = SimpleLinearNet(input_schema,output_schema).to(device)
    elif args.model == 'transformer':
        model = MultidimTransformerModel(input_schema, output_schema).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "multidim_model.pt")


if __name__ == '__main__':
    
    # Define input and output dimensions
    input_schema = DataSchema([
        
        DataDimension("dim1", 490)
        
    ])
    output_schema = DataSchema([
       
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
        DataDimension("dim3", 10)
        
    ])

    main()
