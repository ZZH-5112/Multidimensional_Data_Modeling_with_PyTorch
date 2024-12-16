# Multidimensional Data Modeling Framework (PyTorch)

## Overview

This framework provides a flexible and extensible solution for multidimensional data modeling using PyTorch. It supports datasets with arbitrary dimensions, allowing dynamic input and output definitions. The framework includes features for generating synthetic or CSV-based multidimensional data, training models, and evaluating results. It currently supports simple fully connected neural networks and transformer-based architectures, making it adaptable for a variety of tasks.

## Dependencies

- PyTorch
- NumPy
- argparse
- einops
- transformers

## Installation Requirements

Install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

## Features

- **Dynamic Data Dimensions**: Define input and output data schemas dynamically to support datasets with any number of dimensions.
- **Flexible Dataset Handling**: Supports both synthetic (randomly generated) and CSV-based datasets for training and testing.
- **Multiple Model Options**: Includes a simple fully connected neural network and a transformer-based model.
- **Configurable Training Workflow**: Provides command-line options to configure batch size, learning rate, epochs, and other parameters.
- **Extensibility**: Easily adapt the framework to include custom data dimensions, model architectures, and loss functions.

## Usage

### Command-line Arguments

The framework accepts the following command-line arguments:

| Argument             | Description                                   | Default              |
|----------------------|-----------------------------------------------|----------------------|
| `--batch-size`       | Batch size for training                      | `64`                |
| `--test-batch-size`  | Batch size for testing                       | `1000`              |
| `--epochs`           | Number of training epochs                    | `14`                |
| `--lr`               | Learning rate                                | `0.001`             |
| `--gamma`            | Learning rate step decay factor              | `0.7`               |
| `--no-cuda`          | Disable CUDA (GPU) training                  | `False`             |
| `--dry-run`          | Perform a single forward pass for testing    | `False`             |
| `--seed`             | Random seed                                  | `1`                 |
| `--log-interval`     | Logging interval for training progress       | `10`                |
| `--save-model`       | Save the model after training                | `False`             |
| `--model`            | Model type (`linear` or `transformer`)       | `transformer`       |
| `--dataset`          | Dataset type (`random` or `csv`)             | `random`            |

### Example Commands

#### Train a Transformer Model with Default Settings
```bash
python main.py --epochs 14 --model transformer
```

#### Train a Linear Model with a Custom Batch Size
```bash
python main.py --model linear --batch-size 128 --epochs 20
```

#### Use CSV Data for Training and Testing
```bash
python main.py --dataset csv --model transformer
```

### Example Configuration

#### Modify Input and Output Data Dimensions

To customize the input and output dimensions, modify the `input_schema` and `output_schema` directly in `main.py`:

```python
input_schema = DataSchema([
    DataDimension("dim1", 490)
])

output_schema = DataSchema([
    DataDimension("dim1", 1),
    DataDimension("dim2", 10),
    DataDimension("dim3", 10)
])
```

#### Define a Dataset with These Dimensions

```python
dataset = MultidimDataset(input_schema, output_schema, num_samples=1000)
```

This creates a dataset with 1000 samples, each following the defined input and output schema.

## Workflow

1. **Data Preparation**: Define the input and output schemas and generate synthetic or CSV-based datasets using `MultidimDataset` or `CSVMultidimDataset`.
2. **Model Selection**: Choose between a simple linear model or a transformer-based model.
3. **Training**: Train the model using mean squared error (MSE) loss by default.
4. **Testing**: Evaluate the model on a test set and log the average loss.

## Extensibility

### 1. **Custom Data Schemas**
You can easily adapt the framework to work with different data configurations by modifying the input and output schemas.

#### Example:
```python
input_schema = DataSchema([
    DataDimension("dim1", 24),
    DataDimension("dim2", 5),
    DataDimension("dim3", 10)
])

output_schema = DataSchema([
    DataDimension("dim1", 6),
    DataDimension("dim2", 10)
])
```

### 2. **Add New Models**
Extend the framework by adding new model classes, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs).

### 3. **Change Loss Function**
By default, the framework uses `torch.nn.functional.mse_loss`. You can change it to other loss functions like `torch.nn.functional.cross_entropy` or implement a custom loss function.

### 4. **Custom Preprocessing**
Implement preprocessing steps, such as data normalization or augmentation, within the dataset classes (`MultidimDataset` or `CSVMultidimDataset`).

## Output

During training, the framework logs the loss after each logging interval and evaluates the model's average loss on the test set at the end of each epoch.

#### Example Output:
```
Train Epoch: 1 [0/10000 (0%)]	Loss: 0.046215
Train Epoch: 1 [640/10000 (6%)]	Loss: 0.041732
...
Test set: Average loss: 0.0235
```

## Saving and Loading Models

To save the trained model, use the `--save-model` argument. The model will be saved as `multidim_model.pt`. You can load it later with:

```python
model.load_state_dict(torch.load("multidim_model.pt"))
```

## Conclusion

This framework provides a robust starting point for multidimensional data modeling using PyTorch. With its flexible design, it can be adapted to a wide range of data modeling tasks, from simple regression problems to more complex prediction tasks involving high-dimensional data.