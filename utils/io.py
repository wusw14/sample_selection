import numpy as np
import os
import pandas as pd
from collections import namedtuple
from .prefix import KEY_ATTR

__all__ = ["EntryPair", "load_data"]

EntryPair = namedtuple("EntryPair", ["cols", "valsA", "valsB"])


def read_table(filepath, is_wdc_dataset=False, index_col=None, filter_cols=True):
    if is_wdc_dataset:
        df = pd.read_csv(filepath, header=None, sep="\t")
    else:
        if index_col is not None:
            df = pd.read_csv(filepath, index_col=index_col)
        else:
            df = pd.read_csv(filepath)
    cols = list(df.columns)
    if "table" in filepath and is_wdc_dataset == False and filter_cols:
        dataset_name = filepath.split("/")[-2]
        cols = KEY_ATTR.get(dataset_name, cols)
        df = df[cols]
    return cols, df


def wdc_process(text):
    text = text.replace("COL title VAL ", "")
    text = [" ".join(text.split())]
    return text


def load_data(data_dir, filename, is_wdc_dataset, filter_cols=True):
    entry_pairs, labels = [], []
    filepath = os.path.join(data_dir, filename)
    is_test = "test" in filename
    if is_wdc_dataset:
        _, df = read_table(filepath, is_wdc_dataset=True, filter_cols=False)
        for textA, textB, y in df.values:
            entry_pairs.append(
                EntryPair(["title"], wdc_process(textA), wdc_process(textB))
            )
            labels.append(y)
    else:
        cols, dfA = read_table(
            os.path.join(data_dir, "tableA.csv"), index_col=0, filter_cols=filter_cols
        )
        _, dfB = read_table(
            os.path.join(data_dir, "tableB.csv"), index_col=0, filter_cols=filter_cols
        )
        _, df = read_table(filepath, filter_cols=filter_cols)
        if is_test == False and os.path.exists(os.path.join(data_dir, "black_list.csv")):
            black_list = pd.read_csv(os.path.join(data_dir, "black_list.csv"))
            black_list = black_list["ltable_id"].values
        else:
            black_list = []
        dfA, dfB = dfA.values, dfB.values
        for lid, rid, y in df.values:
            if lid in black_list:
                continue
            entry_pairs.append(EntryPair(cols, dfA[lid], dfB[rid]))
            labels.append(y)
    print(f"Loaded {filename}, original size = {len(df)}, final size = {len(labels)}")
    return entry_pairs, labels
