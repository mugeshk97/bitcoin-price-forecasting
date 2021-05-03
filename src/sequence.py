import numpy as np
from tqdm import tqdm
import pandas as pd



def create_sequence(input_data : pd.DataFrame, sequence_length, target_column, future_length = 1):
    X = []
    y = []
    for i in tqdm(range(sequence_length, len(input_data) - future_length +1)):
        X.append(input_data[i - sequence_length : i, 0: input_data.shape[1]])
        y.append(input_data[i + future_length - 1: i + future_length, target_column])
    return np.array(X), np.array(y)