# Multidimensional Data Modeling Framework (PyTorch)

## Overview

This code provides a flexible framework for multidimensional data modeling using PyTorch. The framework supports handling datasets with arbitrary dimensions, allowing researchers to dynamically define the structure of input and output data. The code includes functionality for generating synthetic multidimensional data, training a simple fully connected neural network model, and performing testing and evaluation. The design of the framework is highly extensible, enabling researchers to adjust data dimensions and model structures based on their task requirements.

## Dependencies

- PyTorch
- NumPy
- argparse
- transformers
- einops

## Installation Requirements

```bash
pip install -r requirements.txt
```

## Features

- **Dynamic Data Dimensions**: Researchers can define the dimensions and sizes of input and output data, supporting modeling with datasets of any number of dimensions.
- **Synthetic Data Generation**: Generates random multidimensional data for training and testing.
- **Simple Neural Network**: Implements a simple fully connected neural network that can be modified as needed.
- **Complete Training and Testing Workflow**: Provides a complete workflow for training, testing, and evaluating, including loss function logging.
- **Highly Extensible**: The framework is easily extensible, supporting more complex network structures or different loss functions.

## Usage

### Command-line Arguments

- `--batch-size`: Batch size during training (default: 64)
- `--test-batch-size`: Batch size during testing (default: 1000)
- `--epochs`: Number of training epochs (default: 14)
- `--lr`: Learning rate (default: 0.01)
- `--gamma`: Learning rate decay factor (default: 0.7)
- `--no-cuda`: Whether to disable CUDA training (default: False)
- `--dry-run`: Quick check, only performs one forward pass (default: False)
- `--seed`: Random seed (default: 1)
- `--log-interval`: Interval for logging (default: 10)
- `--save-model`: Whether to save the model after training (default: False)
- `--model`: The model used for training (default: transformer)

### Example Command

Train the model with default parameters:

```bash
python main.py --epochs 20 --batch-size 128 --lr 0.001 --model transformer
```

### Modify Data Dimensions Example

You can change the input and output dimensions by modifying the `DataSchema`. For example:

```python
input_schema = DataSchema([
    DataDimension("dim1", 10),
    DataDimension("dim2", 10)
])

output_schema = DataSchema([
    DataDimension("dim1", 10),
    DataDimension("dim2", 10)
])

dataset = MultidimDataset(input_schema, output_schema, num_samples=1000)
```

This will generate a dataset of 1000 samples, where each sample contains input and output with the specified dimensions.

### Modify Model Example

You can specify the model used for training with `--model = transformer`. The currently supported models are:

* Transformer

### Model Training and Testing Workflow

1. **Data Generation**: The `MultidimDataset` class is used to generate synthetic multidimensional datasets.
2. **Model Definition**: A simple fully connected neural network model is defined using `MultidimNet`, and its structure is dynamically generated based on the input and output dimensions.
3. **Training**: During training, the model is trained using mean squared error (MSE) loss.
4. **Testing**: After each training epoch, the model is evaluated on the test set, and the average loss is output.

## Extensibility

### 1. **Custom Data Schema**

You can modify the `DataSchema` to generate input and output data with different dimensions.

```python
input_schema = DataSchema([
    DataDimension("dim1", 24),
    DataDimension("dim2", 5),
    DataDimension("dim3", 10),
    DataDimension("dim4", 10)
])

output_schema = DataSchema([
    DataDimension("dim1", 6),
    DataDimension("dim2", 10),
    DataDimension("dim3", 10)
])
```

### 2. **Custom Network Architecture**

The current model is a simple fully connected network. You can extend it into more complex architectures such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) as needed.

### 3. **Loss Function**

You can modify the loss function from `mse_loss` to other types, such as `cross_entropy_loss` or any custom loss function.

### 4. **Data Preprocessing**

You can implement preprocessing steps in the `MultidimDataset` class, such as normalization, data augmentation, or handling missing values.