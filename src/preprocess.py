#from load_data import load_data

import numpy as np
import pandas as pd
import os

distinct_features = ["By Owner",
                     "Exclusive",
                     "Sublet / Lease-Break",
                     "No Fee",
                     "Reduced Fee",
                     "Short Term Allowed",
                     "Furnished",
                     "Laundry In Unit",
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

def add_dummy_features(data):
    print ("Adding dummy features...")
    dist = data.features.apply(
        lambda x: pd.Series(map(lambda z: 1 if (z in x) else 0, distinct_features) +
                            [len(np.setdiff1d(x, distinct_features))]))
    dist.columns = distinct_features + ["unique_count"]

    return data.join(dist)

def add_manager_id_count(data):
    print ("Adding manager count...")
    man_counts = pd.DataFrame(data.manager_id.value_counts())
    man_counts["manager count"] = man_counts["manager_id"]
    man_counts["manager_id"] = man_counts.index

    return pd.merge(data, man_counts, on="manager_id")

def process_data(prefile, postfile):
    """
    Read prefile as json, process it, write the processed data to postfile,
    and return the processed data.
    """
    pre = os.path.basename(prefile)
    post = os.path.basename(postfile)
    print("\nOpening '{}'...".format(pre))
    with open(prefile) as f:
        data = pd.read_json(f)

    print("Pre-processing data...")

    data = add_dummy_features(data)
    data = add_manager_id_count(data)

    print("Finished processing data.")

    print("Writing processed data to '{}'...".format(post))
    with open(postfile, "w") as p:
        data.to_json(p)

    print("Finished processing '{}' into '{}'.".format(pre, post))

    return data

