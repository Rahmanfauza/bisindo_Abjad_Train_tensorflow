
import numpy as np
import os

file_path = "c:/Users/Rahman/Desktop/tensorflow/dataset_bisindo/train/A/A_b01_s01_20251203_183029.npy"
try:
    data = np.load(file_path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
    print(f"Values [0-5]: {data.flatten()[:5]}")
    print(f"Values [63-68]: {data.flatten()[63:68]}")
    print(f"Values [125-131]: {data.flatten()[125:]}")
except Exception as e:
    print(f"Error loading npy: {e}")
