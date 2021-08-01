"""
    Description: Creates the DF used while training the models for task-B in a `(statement, [cells])` format
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
        aid2index = {}
        curr_index = 0
        for s in t.iter("statement"):
            stype = s.attrib["type"]
            if stype != "unknown":
                sid = s.attrib["id"]
                allowed_ids.append(sid)
                aid2index[sid] = curr_index
                stext = s.attrib["text"]
                id2txt[sid] = stext
                curr_index += 1

        cell_text_list = [[]]*curr_index
        relevancy_list = [[]]*curr_index

        # Iterate over the cells:
        for c in t.iter("cell"):

            ctext = c.attrib["text"]

            # Iterate over the evidences present in the cell
            for e in c.iter("evidence"):
                # We consider all versions since they account for the uncertainty in labelling the data
                relevancy = e.attrib["type"]
                eid = e.attrib["statement_id"]
                if eid in allowed_ids:
                    cell_text_list[aid2index[eid]].append(ctext)
                    relevancy_list[aid2index[eid]].append(relevancy2Num[relevancy])

        for eid in allowed_ids :
            df_dict["statement_text"].append(id2txt[eid])
            df_dict["cell_text"].append(str(cell_text_list[aid2index[eid]]))
            df_dict["relevancy"].append(str(relevancy_list[aid2index[eid]]))

if __name__ == "__main__":

    BASE_PATH = sys.argv[1] #"../dataset/train_manual_v1.3.2/v1.3.2/"

    FILE_RE = "*.xml" # RegEx used while scanning the input directory

    # Sorted to ensure order is deterministic, by default glob.glob scans in (ls -U) order
    for fullpath in tqdm(sorted(glob.glob(os.path.join(BASE_PATH+"output/", FILE_RE)))):
        dataBuilder(fullpath)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.astype(str) # To avoid any reading errors
    df.index.name = "id"
    
    # df = df.sample(n = N, random_state = 42)

    SAVE_PATH = f"{BASE_PATH}/data_task_b_list.csv"
    df.to_csv(SAVE_PATH)
