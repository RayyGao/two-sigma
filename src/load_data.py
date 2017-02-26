import numpy as np
import pandas as pd

distinct_features = ["By Owner",
                     "Exclusive",
                     "Sublet / Lease-Break",
                     "No Fee",
                     "Reduced Fee",
                     "Short Term Allowed",
                     "Furnished",
                     "Laundry In Unit"
                     "Private Outdoor Space",
                     "Parking Space",
                     "Cats Allowed",
                     "Dogs Allowed",
                     "Doorman",
                     "Elevator",
                     "Fitness Center",
                     "Laundry In Building",
                     "Common Outdoor Space",
                     "Storage Facility"]


def load_data(filename):
    with open(filename) as f:
        data = pd.read_json(f)

    return data

data = load_data("../data/train.json")

dist = data.features.apply(
    lambda x: pd.Series(map(lambda z: z in x, distinct_features) +
                        [list(np.setdiff1d(x, distinct_features))]))
dist.columns = distinct_features + ["UNIQUES"]
