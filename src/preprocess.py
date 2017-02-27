import numpy as np
import pandas as pd

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


def add_description_analysis(data):
    return data


def process_data(data):
    """
    Read prefile as json, process it, and return the processed data.
    """
    print("Pre-processing data...")

    data = add_dummy_features(data)
    data = add_manager_id_count(data)
    data = add_description_analysis(data)

    print("Finished processing data.")

    return data
