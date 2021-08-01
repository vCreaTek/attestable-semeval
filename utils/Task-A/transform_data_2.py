"""
    Description: 
    Given a (table, statement) DF, generates a (table, statements) DF directly
    Author: Harshit Varma (GitHub: @hrshtv)
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

csv_path  = sys.argv[1] # Input path: "../../dataset/train_manual_v1.3.2/v1.3.2/data.csv"
save_path = sys.argv[2] # Ouput path: "data_new.csv"

df = pd.read_csv(csv_path, index_col = "id")

# The DF which will contain only the unknown statements, will be concatenated with df later 
df_new = {
    "table_name" : [],
    "statements" : [],
    "labels" : []
}

table_names = df["table_name"].unique()
n_tables = len(table_names)

for i, tname in enumerate(tqdm(table_names)):
    mask = (df["table_name"] == tname)
    statements = df["statement"][mask].to_list()
    labels = df["label"][mask].to_list()

    df_new["table_name"].append(tname)
    df_new["statements"].append(statements) 
    df_new["labels"].append(labels)

df_new = pd.DataFrame.from_dict(df_new)
df_new.index.name = "id"
df_new.to_csv(save_path)
    