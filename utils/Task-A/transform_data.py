"""
    Description: 
    Creates the main df with table_name and statement as columns, 
    while simultaneously creating the CSV files for the tables
    This is done simultaneously to avoid any ordering discrepancy
    This file is meant to be for augmented data, with 3 (or 2) classes
    Here, we also incorporate table legend, caption and footer
    A table MAY have a caption, a legend and/or a footnote
    caption : Treated as a single row at the top of the table, with cell.text = caption.text
    legend : Treated as a single row added after the caption, since the model should "know" the legends used before seeing the table. Another thing to try: Add legend both at the start AND at the end
    footnote : Treated as a single row at the end of the table, with cell.text = footnote.text
    We consider a data sample as a (table, statement) pair and not (table, statements)

    Author: Harshit Varma (GitHub: @hrshtv)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Set to True if you want to include table metadata (caption, legend, footer)
INCLUDE_TABLE_META = False

# Global variables
table_names_list = []
statement_list = []
labels = []

def getTableDims(t):
    """ Returns the dimension of table taking into account nested headers etc """
    rows = t.findall("row") # can be used for making the table
    n_rows = len(rows) # Total number of rows in the table, including sub-headers etc
    # Now we need to find n_cols
    max_col = 0
    # Iterate over all cells in the table
    for c in t.iter("cell"):
        col_end = int(c.attrib["col-end"])
        if col_end > max_col:
            max_col = col_end
    n_cols = max_col + 1 # Since 0 based indexing
    return n_rows, n_cols

def xmlToDataframe(t):
    """ Converts given table node to a pandas dataframe """
    n_rows, n_cols = getTableDims(t)

    # Initialize a n_rows x n_cols table
    unk = "[UNK]" # Use whatever unknown token you want here
    table = np.array([[unk]*n_cols]*n_rows, dtype = "object") # Convert to numpy array for easier slicing and setting

    for c in t.iter("cell"):
        col_start = int(c.attrib["col-start"])
        col_end = int(c.attrib["col-end"])
        row_start = int(c.attrib["row-start"])
        row_end = int(c.attrib["row-end"])
        ctext = c.attrib["text"].replace(",", " ") # Replace commas else there will be errors while reading the CSV
        table[row_start:row_end+1, col_start:col_end+1] = ctext # Set entire submatrix to the text present in the cell

    if INCLUDE_TABLE_META:
        # Now, add the table legend, caption and footer
        # Adding legend before caption ensures that caption can be easily added on top of the legend
        legend = t.find("legend")
        if legend != None:
            legend = legend.attrib["text"].replace(",", " ")
            # Add an extra row at the top of the table and fill all cells with the legend
            row_legend = np.array([legend]*n_cols, dtype = "object")
            table = np.vstack((row_legend, table))

        caption = t.find("caption")
        if caption != None:
            caption = caption.attrib["text"].replace(",", " ")
            # Add an extra row at the top of the table (could be above the added legend) and fill all cells with the caption
            row_caption = np.array([caption]*n_cols, dtype = "object")
            table = np.vstack((row_caption, table))

        footnote = t.find("footnote")
        if footnote != None:
            footnote = footnote.attrib["text"].replace(",", " ")
            row_footnote = np.array([footnote]*n_cols, dtype = "object")
            table = np.vstack((table, row_footnote))
            # Add an extra row at the bottom of the table and fill all cells with the footnote

    # Row and column headers are default kept as 0,1,2,3..., this may create problems if TAPAS extracts this seperately via df.columns
    df_table = pd.DataFrame(data = table, index = None, columns = None)
    return df_table

def csvBuilder(base_path, csv_path, filename):
    """ Helps building the main dataframe from the output file """
    outpath = f"{base_path}/{filename}"
    filename_raw = filename.split(".")[0] # Remove the .xml extension

    csv_rel_path = csv_path.split("/")[-1]

    allowed_types = ["entailed", "refuted", "unknown"]
    label_map = {"entailed":1, "refuted":0, "unknown":2}

    tree = ET.parse(outpath) # The element tree
    root = tree.getroot() # Root node

    i = 1 # Index for the table
    for t in root.iter("table"): # Iterate over all tables   

        # Create and save the table as a CSV
        df_table = xmlToDataframe(t) # Convert to dataframe
        tmp_outpath = f"{filename_raw}_{i}.csv"
        df_table.to_csv(
            f"{csv_path}/{tmp_outpath}", 
            header = False, 
            index = False
        )

        # Build the CSV
        for s in t.iter("statement"):
            stype = s.attrib["type"]
            stext = s.attrib["text"].replace(",", " ")
            if stype in allowed_types:
                label = label_map[stype]
                labels.append(label)
                table_names_list.append(f"{csv_rel_path}/{tmp_outpath}")
                statement_list.append(stext)
        i += 1


if __name__ == "__main__":

    BASE_PATH = sys.argv[1] # augmented_dataset/train_manual_v1.3.2/
    CSV_PATH = sys.argv[2] # augmented_dataset/csv_data/csv_manual (don't add the trailing '/')
    DF_SAVE_PATH = sys.argv[3] # augmented_dataset/csv_data/data_manual.csv
    FILE_RE = "*.xml" # RegEx used while scanning the input directory

    if not os.path.exists(f"{CSV_PATH}"):
        os.makedirs(f"{CSV_PATH}")

    # Sorted to ensure order is deterministic, by default glob.glob scans in (ls -U) order
    for fullpath in tqdm(sorted(glob.glob(os.path.join(BASE_PATH, FILE_RE)))):
        filename = fullpath.split("/")[-1]
        csvBuilder(BASE_PATH, CSV_PATH, filename)
    
    df = pd.DataFrame.from_dict({
        "table_name" : table_names_list, 
        "statement" : statement_list,
        "label" : labels
    })

    df.index.name = "id"
    df.to_csv(DF_SAVE_PATH)