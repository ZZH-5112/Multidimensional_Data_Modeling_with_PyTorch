# 多维数据建模框架 (PyTorch)

## 概述

这段代码提供了一个灵活的框架，旨在使用 PyTorch 进行多维数据建模。该框架支持处理具有任意维度的数据集，允许研究者动态定义输入和输出数据的结构。代码包含生成合成多维数据、训练简单的全连接神经网络模型、以及进行测试和评估的功能。框架的设计具有良好的扩展性，研究者可以根据任务需求调整数据维度和模型结构。

## 依赖库

- PyTorch
- NumPy
- argparse
- transformers 
- einops


## 安装要求

```bash
pip install -r requirements.txt
```

## 特性

- **动态数据维度**：研究者可以定义输入和输出数据的维度和大小，支持任意维度的数据建模。
- **合成数据生成**：生成随机的多维数据用于训练和测试。
- **简单神经网络**：实现了一个简单的全连接神经网络，研究者可以根据需要进行修改。
- **完整的训练与测试流程**：提供了完整的训练、测试和评估流程，并包括损失函数日志。
- **可扩展性强**：框架可以轻松扩展，支持更复杂的网络结构或不同的损失函数。

## 使用方法

### 命令行参数

- `--batch-size`：训练时的批量大小（默认：64）
- `--test-batch-size`：测试时的批量大小（默认：1000）
- `--epochs`：训练的轮次（默认：14）
- `--lr`：学习率（默认：0.01）
- `--gamma`：学习率的衰减系数（默认：0.7）
- `--no-cuda`：是否禁用CUDA训练（默认：False）
- `--dry-run`：快速检查，仅进行一次前向传播（默认：False）
- `--seed`：随机种子（默认：1）
- `--log-interval`：日志打印间隔（默认：10）
- `--save-model`：训练后是否保存模型（默认：False）
- `--model`：训练使用的模型（默认：transformer）

### 示例命令

使用默认参数训练模型：

```bash
python main_mutil_dim.py --epochs 20 --batch-size 128 --lr 0.001 --model transformer
```

### 修改数据维度示例

您可以通过修改 `DataSchema` 来更改输入和输出的维度。例如：

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

这将生成 1000 个样本的数据集，每个样本包含具有指定维度的输入和输出。

### 修改模型示例

您可以通过`--model = transformer`来指定训练所使用的模型。目前支持的模型如下所示：

* Transformer

### 模型训练与测试流程

1. **数据生成**：`MultidimDataset` 类用于生成合成的多维数据集。
2. **模型定义**：使用 `MultidimNet` 定义一个简单的全连接神经网络模型，模型结构基于输入和输出的维度动态生成。
3. **训练**：在训练过程中，模型使用均方误差损失（MSE）进行训练。
4. **测试**：每轮训练后，模型会在测试集上进行评估，输出平均损失。

## 可扩展性

### 1. **自定义数据 schema**

您可以通过修改 `DataSchema` 来进行生成的输入输出数据，维度的变化。

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

### 2. **自定义网络架构**

目前的模型是一个简单的全连接网络，您可以根据需求扩展为更复杂的网络架构，例如卷积神经网络（CNN）或循环神经网络（RNN）。

### 3. **损失函数**

您可以将损失函数从 `mse_loss` 修改为其他类型的损失函数，如 `cross_entropy_loss` 或其他自定义损失函数。

### 4. **数据预处理**

您可以在 `MultidimDataset` 类中实现数据的预处理步骤，例如归一化、数据增强或缺失值处理。
