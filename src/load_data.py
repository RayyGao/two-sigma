import numpy as np
import pandas as pd

def load_train():
    with open(filename) as f:
        data = pd.read_json(f)

    return load_data("../data/train.json")


