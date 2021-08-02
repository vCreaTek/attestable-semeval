"""
    Description: 
    Un-augments the DF and removes unknown statements 

    input: CSV containing the table_name, statement, and label columns
    output: Another CSV, with removed "unknown" statements

    Author: Harshit Varma (GitHub: @hrshtv)
"""

import sys
import pandas as pd

csv_path  = sys.argv[1] # Input path: "../../dataset/train_manual_v1.3.2/v1.3.2/data.csv"
save_path = sys.argv[2] # Ouput path: "data_aug.csv"

df = pd.read_csv(csv_path, index_col = "id")

mask = (df["label"] != 2)
df = df[mask]
df.reset_index(drop = True, inplace = True)
df.index.name = "id"
df.to_csv(save_path)