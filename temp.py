from datasets import load_dataset

# Replace 'dataset_name' with the actual name of the dataset you want to download
dataset = load_dataset("Skylion007/openwebtext")

# Access the dataset as needed, e.g., print the first example of the training set
print(dataset['train'][0])