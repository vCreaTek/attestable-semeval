"""
    Description: 
    Merges two given DFs

    Author: Harshit Varma (GitHub: @hrshtv)
"""

import sys
import pandas as pd 

path1 = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]

df1 = pd.read_csv(path1, index_col = 0)
df2 = pd.read_csv(path2, index_col = 0)

df = pd.concat([df1, df2], ignore_index = True)
df.index.name = "id"

df.to_csv(path3)