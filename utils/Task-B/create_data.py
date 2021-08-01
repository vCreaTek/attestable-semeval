"""
    Description: Creates the DF used while training the models for task-B in a `(cell, statement)` format
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

relevancy2Num = {"relevant":1, "irrelevant":0}

# DataFrame to be used for task B
df_dict = {
    "col_start" : [],
    "col_end" : [],
    "row_start" : [],
    "row_end" : [],
    "cell_text" : [],
    "statement_text" : [],
    "relevancy" : []
}

def dataBuilder(outpath):
    """ Helps building the main dataframe for task B from the output file """

    tree = ET.parse(outpath) # The element tree
    root = tree.getroot() # Root node

    for t in root.iter("table"): # Iterate over all tables   

        # Build statement_id:text map
        id2txt = {}
        allowed_ids = []
        for s in t.iter("statement"):
            stype = s.attrib["type"]
            if stype != "unknown":
                sid = s.attrib["id"]
                allowed_ids.append(sid)
                stext = s.attrib["text"]
                id2txt[sid] = stext

        # Iterate over the cells:
        for c in t.iter("cell"):

            col_start = c.attrib["col-start"]
            col_end = c.attrib["col-end"]
            row_start = c.attrib["row-start"]
            row_end = c.attrib["row-end"]
            ctext = c.attrib["text"]

            # Iterate over the evidences present in the cell
            for e in c.iter("evidence"):
                # We consider all versions since they account for the uncertainty in labelling the data
                relevancy = e.attrib["type"]
                eid = e.attrib["statement_id"]
                if eid in allowed_ids:
                    df_dict["col_start"].append(col_start)
                    df_dict["col_end"].append(col_end)
                    df_dict["row_start"].append(row_start)
                    df_dict["row_end"].append(row_end)
                    df_dict["cell_text"].append(ctext)
                    df_dict["statement_text"].append(id2txt[eid])
                    df_dict["relevancy"].append(relevancy2Num[relevancy])

if __name__ == "__main__":

    BASE_PATH = sys.argv[1] #"../dataset/train_manual_v1.3.2/v1.3.2/"
    N = int(sys.argv[2]) # Size of the DF
    FILE_RE = "*.xml" # RegEx used while scanning the input directory

    # Sorted to ensure order is deterministic, by default glob.glob scans in (ls -U) order
    for fullpath in tqdm(sorted(glob.glob(os.path.join(BASE_PATH+"output/", FILE_RE)))):
        dataBuilder(fullpath)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.astype(str) # To avoid any reading errors
    df.index.name = "id"
    
    df = df.sample(n = N, random_state = 42)

    SAVE_PATH = f"{BASE_PATH}/data_task_b_{N}.csv"
    df.to_csv(SAVE_PATH)
