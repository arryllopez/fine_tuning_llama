#imports
import torch 
from datasets import laod_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize dataset
DATASET_NAME = "ChrisHayduk/Llama-2-SQL-Dataset" 
#load the dataset
dataset = load_dataset(DATASET_NAME)
