import os
import pandas as pd

import sys
sys.path.append(r"../")

from base import PROJECT_ROOT
from preprocess import process_data


def save_data(filename):
    print("Creating processed datafile for '{}'".format(filename))

    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', filename)
    PROC_PATH = os.path.join(PROJECT_ROOT, 'data',
                             "processed_" + filename)

    print(DATA_PATH)
    try:
        pre = os.path.basename(DATA_PATH)
        post = os.path.basename(PROC_PATH)

        print("\nOpening '{}'...".format(pre))
        with open(DATA_PATH, 'r') as f:
            data = pd.read_json(f)

        data = process_data(data)

        print("Writing processed data to '{}'...".format(post))

        with open(PROC_PATH, "w") as p:
            data.to_json(p)

        print("Finished processing '{}' into '{}'.".format(pre, post))
    except Exception as e:
        print("Failed to process {} to file.".format(DATA_PATH))
        print(e)

def load_processed_data(dataname):
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data',
                             "processed_" + dataname)
    with open(DATA_PATH) as f:
            data = pd.read_json(f)

    return data
