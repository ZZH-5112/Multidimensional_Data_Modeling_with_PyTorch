import torch
from .schema import DataDimension, DataSchema
from .multidim_dataset import MultidimDataset
from .csv_multidim_dataset import CSVMultidimDataset

def test_multidim_dataset():
    print("\nTesting MultidimDataset...")
    
    # Define input and output schemas
    input_schema = DataSchema([
        DataDimension("dim1", 10),
        DataDimension("dim2", 7),
        DataDimension("dim3", 7),
        DataDimension("dim3", 7)
    ])

    output_schema = DataSchema([
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
        DataDimension("dim3", 10)
    ])

    # Create dataset
    num_samples = 1000
    dataset = MultidimDataset(input_schema, output_schema, num_samples)

    # Test dataset shape
    print(f"Dataset length: {len(dataset)}")
    print(f"Expected input shape: {input_schema.get_shape()}")
    print(f"Expected output shape: {output_schema.get_shape()}")
    
    # Test single sample
    sample_input, sample_output = dataset[0]
    print(f"Actual input shape: {sample_input.shape}")
    print(f"Actual output shape: {sample_output.shape}")

    assert len(dataset) == num_samples, f"Expected {num_samples} samples, but got {len(dataset)}"
    assert sample_input.shape == torch.Size(input_schema.get_shape()), f"Expected input shape {input_schema.get_shape()}, but got {sample_input.shape}"
    assert sample_output.shape == torch.Size(output_schema.get_shape()), f"Expected output shape {output_schema.get_shape()}, but got {sample_output.shape}"

    print("MultidimDataset test passed!")

def test_csv_multidim_dataset(csv_file_path):
    print("\nTesting CSVMultidimDataset...")
    
    # Define input and output schemas
    input_schema = DataSchema([
        DataDimension("dim1", 7),
        DataDimension("dim2", 70),
        
    ])

    output_schema = DataSchema([
        DataDimension("dim1", 1),
        DataDimension("dim2", 10),
        DataDimension("dim3", 10)
    ])

    # Create dataset
    dataset = CSVMultidimDataset(csv_file_path, input_schema, output_schema)

    # Test dataset shape
    print(f"Dataset length: {len(dataset)}")
    print(f"Expected input shape: {input_schema.get_shape()}")
    print(f"Expected output shape: {output_schema.get_shape()}")
    
    # Test single sample
    sample_input, sample_output = dataset[0]
    print(f"Actual input shape: {sample_input.shape}")
    print(f"Actual output shape: {sample_output.shape}")

    assert sample_input.shape == torch.Size(input_schema.get_shape()), f"Expected input shape {input_schema.get_shape()}, but got {sample_input.shape}"
    assert sample_output.shape == torch.Size(output_schema.get_shape()), f"Expected output shape {output_schema.get_shape()}, but got {sample_output.shape}"

    print("CSVMultidimDataset test passed!")

if __name__ == "__main__":
    test_multidim_dataset()
    
    # You need to provide the path to your CSV file
    csv_file_path = "Data/sample.csv"
    test_csv_multidim_dataset(csv_file_path)